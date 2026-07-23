"""Likelihood-based multiple-choice evaluation for picochat checkpoints.

Each benchmark item is rendered into one (context, completion) text pair per
answer choice; the model scores each completion's token log-likelihood given
its context, and the prediction is the argmax over choices. Metrics follow
lm-evaluation-harness conventions:

- acc:      argmax over the raw sum of completion-token log-probs
- acc_norm: argmax over that sum normalized by completion byte length

Two prompt renderings share the same items, so numbers are comparable before
and after SFT:

- base (pretrain-only) checkpoints: `<|begin_of_text|>` + context, the
  completion scored as a plain text continuation.
- chat (SFT) checkpoints: the context becomes a ChatML user turn and the
  completion is scored as the start of the assistant reply.

Tasks are HF datasets that ship data files (datasets>=5 no longer runs
script-based datasets, which rules out e.g. piqa).
"""

import re
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from picochat.tokenizer import (
    BOS_TOKEN,
    PAD_TOKEN,
    Tokenizer as Encoding,
    render_chat_prompt,
)


@dataclass(frozen=True)
class MCExample:
    """One multiple-choice item: a (context, completion) text pair per choice
    (contexts usually coincide, but may differ per choice -- winogrande), and
    the index of the correct choice."""

    choices: list[tuple[str, str]]
    answer: int


# ---------------------------------------------------------------------------
# Task formatters: one HF record -> MCExample (None drops malformed records)
# ---------------------------------------------------------------------------

_BRACKETED = re.compile(r"\[.*?\]")


def _clean_hellaswag(text: str) -> str:
    # The wikihow-derived text keeps "[title]"/"[header]" artifacts; strip
    # them the same way lm-evaluation-harness does.
    text = text.strip().replace(" [title]", ". ")
    return _BRACKETED.sub("", text).replace("  ", " ")


def format_hellaswag(doc: dict) -> MCExample:
    ctx = _clean_hellaswag(doc["activity_label"] + ": " + doc["ctx"])
    return MCExample(
        choices=[(ctx, " " + _clean_hellaswag(end)) for end in doc["endings"]],
        answer=int(doc["label"]),
    )


def format_arc(doc: dict) -> MCExample | None:
    labels = list(doc["choices"]["label"])
    if doc["answerKey"] not in labels:
        return None
    ctx = "Question: " + doc["question"] + "\nAnswer:"
    return MCExample(
        choices=[(ctx, " " + text) for text in doc["choices"]["text"]],
        answer=labels.index(doc["answerKey"]),
    )


def format_openbookqa(doc: dict) -> MCExample | None:
    labels = list(doc["choices"]["label"])
    if doc["answerKey"] not in labels:
        return None
    # question_stem is a sentence to complete, not a question -- each choice
    # is scored as its direct continuation.
    return MCExample(
        choices=[(doc["question_stem"], " " + t) for t in doc["choices"]["text"]],
        answer=labels.index(doc["answerKey"]),
    )


def format_winogrande(doc: dict) -> MCExample:
    # Partial scoring: substitute each option for the blank, score the shared
    # remainder of the sentence as the completion.
    cut = doc["sentence"].index("_")
    tail = doc["sentence"][cut + 1 :]
    return MCExample(
        choices=[
            (doc["sentence"][:cut] + option, tail)
            for option in (doc["option1"], doc["option2"])
        ],
        answer=int(doc["answer"]) - 1,
    )


def format_boolq(doc: dict) -> MCExample:
    ctx = f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"
    return MCExample(choices=[(ctx, " no"), (ctx, " yes")], answer=int(doc["answer"]))


# --- Japanese tasks --------------------------------------------------------
# The tokenizer is CJK-tuned, so the ladder should be scored on Japanese too.
# All three ship plain data files (datasets>=5 compatible) under permissive
# terms: JCommonsenseQA (JGLUE, via the sbintuitions parquet mirror) and
# Belebele are CC-BY-SA 4.0; XWinograd jp derives from public Winograd
# schema collections. Japanese completions carry no leading space -- the
# scripts are unspaced, and the tokenizer's pre-tokenizer splits CJK from a
# preceding "答え:" on the script boundary by itself.


def format_jcommonsenseqa(doc: dict) -> MCExample:
    ctx = f"質問: {doc['question']}\n答え:"
    return MCExample(
        choices=[(ctx, doc[f"choice{i}"]) for i in range(5)],
        answer=int(doc["label"]),
    )


def format_belebele(doc: dict) -> MCExample:
    # FLORES passage + comprehension question, 4 choices (1-based gold).
    ctx = f"{doc['flores_passage']}\n質問: {doc['question']}\n答え:"
    return MCExample(
        choices=[(ctx, doc[f"mc_answer{i}"]) for i in range(1, 5)],
        answer=int(doc["correct_answer_num"]) - 1,
    )


@dataclass(frozen=True)
class TaskSpec:
    path: str  # HF hub dataset id
    name: str | None  # HF config name
    split: str
    format: Callable[[dict], MCExample | None]


TASKS: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec("Rowan/hellaswag", None, "validation", format_hellaswag),
    "arc_easy": TaskSpec("allenai/ai2_arc", "ARC-Easy", "test", format_arc),
    "arc_challenge": TaskSpec("allenai/ai2_arc", "ARC-Challenge", "test", format_arc),
    "openbookqa": TaskSpec("allenai/openbookqa", "main", "test", format_openbookqa),
    "winogrande": TaskSpec(
        "allenai/winogrande", "winogrande_xl", "validation", format_winogrande
    ),
    "boolq": TaskSpec("google/boolq", None, "validation", format_boolq),
    # Japanese (see the formatter comment): commonsense QA, reading
    # comprehension, and coreference. xwinograd shares winogrande's schema
    # (sentence with a "_" blank, two options), so its formatter is reused --
    # the partial-scoring split works unchanged on Japanese text.
    "jcommonsenseqa": TaskSpec(
        "sbintuitions/JCommonsenseQA", None, "validation", format_jcommonsenseqa
    ),
    "belebele_ja": TaskSpec("facebook/belebele", "jpn_Jpan", "test", format_belebele),
    "xwinograd_ja": TaskSpec("Muennighoff/xwinograd", "jp", "test", format_winogrande),
}


def load_task_examples(task: str, limit: int | None = None) -> list[MCExample]:
    # Imported lazily so that scoring/formatting stays usable offline and the
    # module import stays light for tests.
    from datasets import load_dataset

    spec = TASKS[task]
    ds = load_dataset(spec.path, spec.name, split=spec.split)
    examples: list[MCExample] = []
    for doc in ds:
        ex = spec.format(doc)
        if ex is not None:
            examples.append(ex)
        if limit is not None and len(examples) >= limit:
            break
    return examples


# ---------------------------------------------------------------------------
# Tokenization and scoring
# ---------------------------------------------------------------------------


def encode_choice(
    tokenizer: Encoding,
    context: str,
    completion: str,
    chat: bool = False,
    max_len: int = 4096,
) -> tuple[list[int], int]:
    """Token ids for one choice and the index where the completion begins --
    positions cont_start..end are the ones scored. Text is encoded with
    encode_ordinary (same as SFT bodies), so it can never resolve to a
    special token."""
    if chat:
        # ChatML: the context is a user turn, the completion opens the
        # assistant reply. The assistant header already ends with "\n", so
        # the leading space the plain-text rendering relies on is dropped.
        ctx_ids = render_chat_prompt([{"role": "user", "content": context}], tokenizer)
        cont_ids = tokenizer.encode_ordinary(completion.lstrip() or completion)
    else:
        # BPE can merge across the context/completion boundary, so recover
        # the completion ids by slicing the full encoding rather than
        # encoding the completion alone. Trailing context whitespace is moved
        # onto the completion first ("... " + "yes" scores like
        # "..." + " yes"), matching lm-evaluation-harness.
        moved = context[len(context.rstrip()) :]
        context, completion = context.rstrip(), moved + completion
        bos = tokenizer.encode_single_token(BOS_TOKEN)
        ctx_ids = [bos, *tokenizer.encode_ordinary(context)]
        full_ids = [bos, *tokenizer.encode_ordinary(context + completion)]
        cont_ids = full_ids[len(ctx_ids) :]
        if not cont_ids:  # degenerate merge swallowed the whole completion
            cont_ids = tokenizer.encode_ordinary(completion)

    if len(ctx_ids) + len(cont_ids) > max_len:
        # Drop the oldest context tokens, keeping the leading BOS/<|im_start|>
        # and the whole completion.
        keep = max(max_len - len(cont_ids) - 1, 1)
        ctx_ids = [ctx_ids[0], *ctx_ids[len(ctx_ids) - keep :]]
    return ctx_ids + cont_ids, len(ctx_ids)


@torch.no_grad()
def score_requests(
    model: torch.nn.Module,
    requests: list[tuple[list[int], int]],
    pad_id: int,
    batch_size: int = 16,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Sum of completion-token log-probs for each (ids, cont_start) request.
    Rows are length-sorted into right-padded batches to minimize padding;
    causal attention keeps the padding from influencing scored positions."""
    scores = [0.0] * len(requests)
    order = sorted(range(len(requests)), key=lambda i: len(requests[i][0]))
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        rows = [requests[i] for i in chunk]
        width = max(len(ids) for ids, _ in rows)
        x = torch.full((len(rows), width), pad_id, dtype=torch.long)
        for r, (ids, _) in enumerate(rows):
            x[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        logprobs = F.log_softmax(model(x.to(device)).float(), dim=-1)
        for r, (req_idx, (ids, cont_start)) in enumerate(zip(chunk, rows)):
            targets = torch.tensor(ids[cont_start:], device=logprobs.device)
            picked = logprobs[r, cont_start - 1 : len(ids) - 1].gather(
                1, targets[:, None]
            )
            scores[req_idx] = float(picked.sum())
    return scores


def summarize(examples: list[MCExample], scores: list[float]) -> dict:
    """Fold flat per-choice scores (in example order) into task metrics."""
    n_correct = n_correct_norm = 0
    inv_choices = 0.0
    pos = 0
    for ex in examples:
        n = len(ex.choices)
        raw = scores[pos : pos + n]
        norm = [
            s / max(len(completion.encode("utf-8")), 1)
            for s, (_, completion) in zip(raw, ex.choices)
        ]
        n_correct += raw.index(max(raw)) == ex.answer
        n_correct_norm += norm.index(max(norm)) == ex.answer
        inv_choices += 1 / n
        pos += n
    count = len(examples)
    return {
        "n": count,
        "acc": n_correct / count,
        "acc_norm": n_correct_norm / count,
        "random": inv_choices / count,
    }


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: Encoding,
    task: str,
    chat: bool = False,
    limit: int | None = None,
    batch_size: int = 16,
    max_len: int = 4096,
    device: torch.device | str = "cpu",
) -> dict:
    """Run one task end to end: load, render, score, aggregate."""
    examples = load_task_examples(task, limit)
    requests = [
        encode_choice(tokenizer, ctx, cont, chat=chat, max_len=max_len)
        for ex in examples
        for ctx, cont in ex.choices
    ]
    scores = score_requests(
        model,
        requests,
        pad_id=tokenizer.encode_single_token(PAD_TOKEN),
        batch_size=batch_size,
        device=device,
    )
    return {"task": task, **summarize(examples, scores)}
