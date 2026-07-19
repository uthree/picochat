"""Tokenized training data: packing, on-disk formats, Datasets, samplers and
the LightningDataModule (nanochat's dataloader.py analogue; the raw HF
sources live in picochat.dataset).

Everything here packs variable-length token sequences into fixed-length rows
instead of padding each one on its own (MosaicBERT-style sequence packing,
https://arxiv.org/abs/2312.17482; greedy best-fit via pack_bins):

- Pretraining: pack_docs packs whole <|begin_of_text|>doc<|end_of_text|>
  documents into rows of block_size+1 tokens, and ShardWriter splits the rows
  across fixed-size shard files (00000.bin, 00001.bin, ...) in one directory
  so no single file grows with the corpus; meta.json records the block_size
  the rows were packed with. PackedDataset returns one row per item, and the
  block_size+1 length lets GPT._loss shift by one internally to compute the
  next-token prediction loss (effective context block_size). A row usually
  holds several documents, each starting with <|begin_of_text|>; GPT._loss
  derives per-token document ids from those markers so attention never
  crosses a document boundary (see Transformer.forward).

- SFT: pack_examples packs (input_ids, labels) conversations encoded by
  picochat.tokenizer.encode_conversation into fixed-length rows plus explicit
  per-token doc_ids; SFTDataset builds them in memory, SFTTensorDataset reads
  the .pt bundle scripts/sft_setup.py writes.
"""

import bisect
import json
from pathlib import Path

import lightning as L
import numpy as np
import tiktoken
import torch
from torch import Tensor
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)

from picochat.tokenizer import encode_conversation


def pack_bins(lengths: list[int], max_length: int) -> list[list[int]]:
    """Assign items of the given lengths to bins of capacity max_length.

    Greedy best-fit over a length histogram: seed each bin with the longest
    unplaced item, then keep filling it with the largest item that still fits.
    Returns one list of item indices per bin; every index appears exactly once.
    """
    by_len: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        assert 0 < n <= max_length
        by_len.setdefault(n, []).append(i)
    sizes = sorted(by_len)  # ascending, for bisect

    def pop_largest_at_most(room: int) -> int | None:
        j = bisect.bisect_right(sizes, room) - 1
        if j < 0:
            return None
        n = sizes[j]
        idx = by_len[n].pop()
        if not by_len[n]:
            del by_len[n]
            sizes.pop(j)
        return idx

    bins: list[list[int]] = []
    while sizes:
        room = max_length
        packed: list[int] = []
        while (idx := pop_largest_at_most(room)) is not None:
            packed.append(idx)
            room -= lengths[idx]
        bins.append(packed)
    return bins


# 32-bit token ids: leaves headroom for vocab beyond 65535 (e.g. up to 128k).
# Writer and reader share this so the two never diverge.
DTYPE = np.uint32

# Default shard size: 2**28 uint32 tokens = 1 GiB per shard file.
DEFAULT_SHARD_TOKENS = 256 * 2**20


class ShardWriter:
    """Splits a continuous token stream across fixed-size shard files.

    Writes 00000.bin, 00001.bin, ... under `out_dir`, each holding at most
    `shard_tokens` tokens. Any *.bin already in `out_dir` is deleted first --
    the sharded equivalent of truncating a single output file with mode "wb";
    shards left over from a previous, longer run would otherwise still be
    picked up as valid data by PackedDataset.
    """

    def __init__(self, out_dir: str | Path, shard_tokens: int = DEFAULT_SHARD_TOKENS):
        assert shard_tokens > 0
        self.out_dir = Path(out_dir)
        self.shard_tokens = shard_tokens
        self.n_shards = 0
        self._in_shard = 0
        self._file = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.out_dir.glob("*.bin"):
            stale.unlink()

    def write(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=DTYPE)
        while tokens.size:
            if self._file is None:  # shards are created lazily: none are empty
                self._file = open(self.out_dir / f"{self.n_shards:05d}.bin", "wb")
                self.n_shards += 1
                self._in_shard = 0
            room = self.shard_tokens - self._in_shard
            head, tokens = tokens[:room], tokens[room:]
            head.tofile(self._file)
            self._in_shard += head.size
            if self._in_shard >= self.shard_tokens:
                self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# Sidecar written next to the shards; records the block_size the rows were
# packed with so PackedDataset can reject a mismatched training config.
META_FILE = "meta.json"


def write_meta(out_dir: str | Path, block_size: int) -> None:
    (Path(out_dir) / META_FILE).write_text(json.dumps({"block_size": block_size}))


def read_meta(shard_dir: str | Path) -> dict:
    path = Path(shard_dir) / META_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"no {META_FILE} in {shard_dir} -- the corpus predates the "
            "packed-row format; re-run scripts/base_setup.py"
        )
    return json.loads(path.read_text())


def pack_docs(
    docs: list[list[int]], block_size: int, pad_id: int, bos_id: int
) -> np.ndarray:
    """Pack whole documents (each already <|begin_of_text|>...<|end_of_text|>)
    into fixed-length rows of block_size+1 tokens, several documents per row
    (MosaicBERT-style sequence packing; greedy best-fit via pack_bins). Room
    that nothing fits into is padded with pad_id. Returns (n_rows,
    block_size+1) in DTYPE.

    A document longer than one row is split into row-sized chunks first, each
    continuation chunk prefixed with bos_id: GPT._loss derives document ids
    from the <|begin_of_text|> markers, so the prefix makes a continuation its
    own document instead of silently merging with whatever precedes it in the
    row it lands in.
    """
    row_len = block_size + 1
    chunks: list[list[int]] = []
    for doc in docs:
        chunks.append(doc[:row_len])
        for i in range(row_len, len(doc), row_len - 1):
            chunks.append([bos_id, *doc[i : i + row_len - 1]])
    bins = pack_bins([len(c) for c in chunks], row_len)
    rows = np.full((len(bins), row_len), pad_id, dtype=DTYPE)
    for r, packed in enumerate(bins):
        pos = 0
        for idx in packed:
            chunk = chunks[idx]
            rows[r, pos : pos + len(chunk)] = chunk
            pos += len(chunk)
    return rows


class PackedDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024):
        """
        Args:
            path: corpus produced by base_setup.py -- a directory of *.bin
                shards holding packed rows, plus a meta.json recording the
                block_size they were packed with.
            block_size: effective context length; must match the corpus's
                meta.json. Each sample is one packed row of block_size+1 tokens.
        """
        self.block_size = block_size
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"no shard directory at {path}")
        meta = read_meta(p)
        if meta["block_size"] != block_size:
            raise ValueError(
                f"{path} was packed with block_size {meta['block_size']} but "
                f"the training config asks for {block_size}; re-run "
                "scripts/base_setup.py with the matching block_size"
            )
        shard_paths = sorted(p.glob("*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"no .bin shards in directory {path}")
        self.paths = [str(q) for q in shard_paths]
        row_len = block_size + 1
        itemsize = np.dtype(DTYPE).itemsize
        counts = []
        for q in shard_paths:
            n = q.stat().st_size // itemsize
            assert n % row_len == 0, (
                f"{q} holds {n} tokens, not a multiple of block_size+1 "
                f"({row_len}); the shards don't match {META_FILE} -- re-run "
                "scripts/base_setup.py"
            )
            counts.append(n // row_len)
        self.n_tokens = int(sum(counts)) * row_len
        self._cum_counts = np.cumsum(counts)
        # Memmaps are opened lazily per DataLoader worker process: one opened
        # in __init__ would share its file descriptor across the worker fork.
        self._mmaps: list[np.memmap | None] = [None] * len(self.paths)

    def _shard(self, i: int) -> np.memmap:
        if self._mmaps[i] is None:
            self._mmaps[i] = np.memmap(self.paths[i], dtype=DTYPE, mode="r")
        return self._mmaps[i]

    def __len__(self) -> int:
        return int(self._cum_counts[-1])

    def __getitem__(self, idx: int) -> Tensor:
        shard = int(np.searchsorted(self._cum_counts, idx, side="right"))
        local = int(idx - (self._cum_counts[shard - 1] if shard else 0))
        start = local * (self.block_size + 1)
        chunk = self._shard(shard)[start : start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int64))


# Number of indices a chunked sampler draws per iteration. Bounds peak memory:
# both samplers below are asked for num_samples == len(train_ds) == the whole
# row count (millions for a large corpus). Producing all of them at once --
# num_samples-sized tensors plus a num_samples-long Python list from .tolist()
# -- costs tens of GB before the first batch is even yielded, which is what
# exhausts host memory and gets the process OOM-killed. Drawing in fixed-size
# chunks keeps peak memory O(_SAMPLE_CHUNK) regardless of corpus size; training
# stops at max_steps long before an "epoch" of num_samples is consumed.
_SAMPLE_CHUNK = 1 << 20


def _sampler_generator(sampler) -> torch.Generator | None:
    """The RNG a chunked sampler draws from. seed=None -> the global RNG
    (single-process convenience). With a seed, a torch.Generator created on
    first use and kept across __iter__ calls, so a re-iterated "epoch"
    continues the stream instead of replaying it. Created lazily (not in
    __init__) so a sampler pickled into a spawned DDP process seeds its
    generator there, not in the parent."""
    if sampler.seed is None:
        return None
    if sampler._generator is None:
        sampler._generator = torch.Generator()
        sampler._generator.manual_seed(sampler.seed)
    return sampler._generator


class GroupWeightedIndexSampler(Sampler[int]):
    """Draws indices from a ConcatDataset's groups with replacement so each
    group's total sampling mass matches its configured weight, regardless of
    the group's size.

    `torch.utils.data.WeightedRandomSampler` would do this by materializing
    one weight per example and drawing via `torch.multinomial`, which (a)
    refuses more than 2**24 (~16.7M) categories -- easily exceeded once
    several pretraining corpora are concatenated -- and (b) at the scale of
    billions of examples, a fp32/fp64 cumulative-weight table loses enough
    precision that many adjacent examples become unreachable. Sampling in two
    stages -- pick a group via a tiny multinomial (one category per group),
    then a uniform offset within it -- needs O(num_groups) memory and has
    neither problem.
    """

    def __init__(
        self,
        group_sizes: list[int],
        group_weights: list[float],
        num_samples: int,
        seed: int | None = None,
    ):
        self.num_samples = num_samples
        self.group_weights = torch.as_tensor(group_weights, dtype=torch.double)
        sizes = torch.as_tensor(group_sizes, dtype=torch.long)
        self.group_sizes = sizes
        self.group_offsets = torch.cumsum(sizes, dim=0) - sizes
        self.seed = seed
        self._generator: torch.Generator | None = None

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = _sampler_generator(self)
        remaining = self.num_samples
        while remaining > 0:
            n = min(_SAMPLE_CHUNK, remaining)
            group_ids = torch.multinomial(
                self.group_weights, n, replacement=True, generator=g
            )
            local = (
                torch.rand(n, dtype=torch.double, generator=g)
                * self.group_sizes[group_ids]
            ).long()
            idx = self.group_offsets[group_ids] + local
            yield from idx.tolist()
            remaining -= n


class UniformIndexSampler(Sampler[int]):
    """Draws indices uniformly at random with replacement, in fixed-size chunks.

    Used for the unweighted train path in place of DataLoader(shuffle=True),
    whose RandomSampler materializes a full `torch.randperm(len)` (then
    `.tolist()`) up front. `len` is the whole row count for a large corpus,
    so that eager permutation exhausts host memory exactly like the weighted
    sampler did (see _SAMPLE_CHUNK / GroupWeightedIndexSampler). Sampling with
    replacement is fine here: training stops at max_steps well before
    num_samples is drawn.
    """

    def __init__(self, dataset_len: int, num_samples: int, seed: int | None = None):
        self.dataset_len = dataset_len
        self.num_samples = num_samples
        self.seed = seed
        self._generator: torch.Generator | None = None

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = _sampler_generator(self)
        remaining = self.num_samples
        while remaining > 0:
            n = min(_SAMPLE_CHUNK, remaining)
            idx = torch.randint(self.dataset_len, (n,), generator=g)
            yield from idx.tolist()
            remaining -= n


class PretrainDataModule(L.LightningDataModule):
    """LightningDataModule over a train dataset and an optional val dataset,
    used by both the pretraining and SFT training scripts (the datasets differ,
    the loader/sampler wiring is the same).

    Multi-GPU: the Trainer must be built with `use_distributed_sampler=False`
    (the training scripts do this). Lightning's default would wrap the chunked
    samplers above in its DistributedSamplerWrapper, which materializes the
    whole len(train_ds)-long index stream as a Python list on every rank --
    exactly the host-memory blowup the chunked samplers exist to avoid.
    Sharding isn't needed anyway: the samplers draw IID with replacement, so
    each rank drawing from its own per-rank seed (`seed + rank`, see
    train_dataloader) is statistically equivalent. The val loader, which does
    want each example visited once, gets a standard DistributedSampler here.
    """

    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset | None,
        batch_size: int,
        num_workers: int = 4,
        train_group_weights: list[float] | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            train_group_weights: one weight per group in `train_ds` (which
                must be a ConcatDataset of those groups), sized so each
                group's total sampling mass equals its configured weight
                regardless of its example count. None -> plain uniform
                shuffling.
            seed: base seed for the train sampler; rank r draws from seed + r
                (identical streams across ranks would give every rank the same
                batches, silently dividing the effective batch by world_size).
                None -> the global RNG, single-process runs only.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_group_weights = train_group_weights
        self.seed = seed
        if train_group_weights is not None:
            assert isinstance(train_ds, ConcatDataset), (
                "train_group_weights requires train_ds to be a ConcatDataset"
            )
        if val_ds is None:
            # Shadow the class method: Lightning's is_overridden() check treats
            # an instance attribute of None as "hook not provided".
            self.val_dataloader = None  # type: ignore[assignment]

    @staticmethod
    def _dist_info() -> tuple[int, int]:
        """(rank, world_size), (0, 1) outside distributed runs. Read at
        dataloader-build time: Lightning calls the *_dataloader hooks after the
        process group is initialized on every rank."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _sampler_seed(self) -> int | None:
        if self.seed is None:
            return None
        return self.seed + self._dist_info()[0]

    def train_dataloader(self) -> DataLoader:
        if self.train_group_weights is not None:
            group_sizes = [len(d) for d in self.train_ds.datasets]
            sampler = GroupWeightedIndexSampler(
                group_sizes,
                self.train_group_weights,
                num_samples=len(self.train_ds),
                seed=self._sampler_seed(),
            )
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
                drop_last=True,
            )
        # Not shuffle=True: DataLoader's RandomSampler would build a full
        # torch.randperm(len(train_ds)) up front, which OOMs on a large corpus
        # for the same reason the weighted path did. UniformIndexSampler draws
        # the same uniform indices lazily in bounded-size chunks.
        sampler = UniformIndexSampler(
            len(self.train_ds),
            num_samples=len(self.train_ds),
            seed=self._sampler_seed(),
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        # Unlike training (IID draws), validation wants each example counted
        # once, so under DDP the set is sharded with a standard
        # DistributedSampler (built here because use_distributed_sampler is
        # False, see the class docstring). It pads the last batch to keep
        # ranks in step, so the aggregated val_loss can count a few examples
        # twice -- the standard, slightly-approximate trade.
        _, world_size = self._dist_info()
        sampler = (
            DistributedSampler(self.val_ds, shuffle=False)
            if world_size > 1
            else None
        )
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# SFT chat data: packed (input_ids, labels, doc_ids) tensors for SFT training.
# The ChatML rendering/loss-masking itself lives with the tokenizer (see
# picochat.tokenizer.encode_conversation).
# ---------------------------------------------------------------------------


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    max_length: int,
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack variable-length (input_ids, labels) examples into fixed-length
    sequences, several examples per sequence, instead of padding each one to
    max_length on its own (MosaicBERT-style sequence packing, see pack_bins
    for the bin-assignment algorithm). Room that nothing fits into is padded
    with pad_id.

    Returns (input_ids, labels, doc_ids), each (n_bins, max_length) int64.
    doc_ids numbers the examples within a bin, with the padding tail getting
    its own final id, so attention can be confined to one example (see
    Transformer.forward). Each example's first-token label is forced to pad_id:
    after the loss shift it would be predicted from the *previous* example's
    last token, which the document mask hides (at a sequence start it was
    never a target to begin with).
    """
    bins = pack_bins([len(ids) for ids, _ in examples], max_length)

    input_ids = torch.full((len(bins), max_length), pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, pad_id)
    doc_ids = torch.zeros_like(input_ids)
    for b, packed in enumerate(bins):
        pos = 0
        for d, idx in enumerate(packed):
            ids, labs = examples[idx]
            end = pos + len(ids)
            input_ids[b, pos:end] = torch.tensor(ids)
            labels[b, pos:end] = torch.tensor(labs)
            labels[b, pos] = pad_id  # never a cross-example target (see above)
            doc_ids[b, pos:end] = d
            pos = end
        doc_ids[b, pos:] = len(packed)  # padding tail: its own document
    return input_ids, labels, doc_ids


class SFTDataset(Dataset):
    """Pre-tokenizes and packs every conversation once at construction time
    (SFT corpora fit in memory, unlike pretraining's token-stream shards) into
    fixed-length (input_ids, labels, doc_ids) sequences, several conversations
    per sequence (see pack_examples)."""

    def __init__(
        self,
        conversations: list[list[dict]],
        tokenizer: tiktoken.Encoding,
        max_length: int,
        pad_id: int,
    ):
        self.pad_id = pad_id
        examples = [
            encoded
            for messages in conversations
            if (encoded := encode_conversation(messages, tokenizer, max_length, pad_id))
            is not None
        ]
        self.input_ids, self.labels, self.doc_ids = pack_examples(
            examples, max_length, pad_id
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "doc_ids": self.doc_ids[idx],
        }


class SFTTensorDataset(Dataset):
    """Reads a (input_ids, labels, doc_ids) tensor bundle written by
    scripts/sft_setup.py's `process()` -- the on-disk counterpart of
    SFTDataset, for training runs that shouldn't re-tokenize on every launch.
    """

    def __init__(self, path: str | Path):
        bundle = torch.load(path, map_location="cpu")
        self.input_ids = bundle["input_ids"]
        self.labels = bundle["labels"]
        self.doc_ids = bundle["doc_ids"]
        self.pad_id = bundle["pad_id"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "doc_ids": self.doc_ids[idx],
        }
