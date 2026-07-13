"""Greedy best-fit bin packing, shared by pretraining (pretrain.pack_docs) and
SFT (sft.pack_examples): both pack variable-length token sequences into
fixed-length rows instead of padding each one on its own (MosaicBERT-style
sequence packing, https://arxiv.org/abs/2312.17482)."""

import bisect


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
