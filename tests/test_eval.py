import torch
import torch.nn.functional as F

from picochat.eval import (
    MCExample,
    TASKS,
    encode_choice,
    format_arc,
    format_boolq,
    format_hellaswag,
    format_openbookqa,
    format_winogrande,
    score_requests,
    summarize,
)
from picochat.model.gpt import TransformerLM
from picochat.tokenizer import BOS_TOKEN, IM_START, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Task formatters
# ---------------------------------------------------------------------------
def test_format_hellaswag_cleans_artifacts():
    doc = {
        "activity_label": "Removing ice from car",
        "ctx": "The man works on the car. [header] He",
        "endings": ["scrapes the ice [substeps] off.", "eats lunch.", "a.", "b."],
        "label": "0",
    }
    ex = format_hellaswag(doc)
    assert ex.answer == 0
    ctx, cont = ex.choices[0]
    assert "[" not in ctx and "[" not in cont
    assert ctx.startswith("Removing ice from car: ")
    assert cont.startswith(" ")  # completions score as " "-prefixed continuations


def test_format_arc_maps_answer_key():
    doc = {
        "question": "Which is a mammal?",
        "choices": {"text": ["snake", "whale", "trout"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    ex = format_arc(doc)
    assert ex.answer == 1
    assert ex.choices[1] == ("Question: Which is a mammal?\nAnswer:", " whale")


def test_format_arc_numeric_labels_and_missing_key():
    doc = {
        "question": "q",
        "choices": {"text": ["x", "y"], "label": ["1", "2"]},
        "answerKey": "2",
    }
    assert format_arc(doc).answer == 1
    assert format_arc({**doc, "answerKey": "Z"}) is None


def test_format_openbookqa_scores_direct_continuation():
    doc = {
        "question_stem": "The sun is responsible for",
        "choices": {"text": ["plants sleeping", "the seasons"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    ex = format_openbookqa(doc)
    assert ex.answer == 1
    assert ex.choices[0] == ("The sun is responsible for", " plants sleeping")


def test_format_winogrande_substitutes_blank():
    doc = {
        "sentence": "The trophy didn't fit because _ was too big.",
        "option1": "the trophy",
        "option2": "the suitcase",
        "answer": "1",
    }
    ex = format_winogrande(doc)
    assert ex.answer == 0
    # per-choice contexts, shared completion (the text after the blank)
    assert ex.choices[0][0] == "The trophy didn't fit because the trophy"
    assert ex.choices[1][0] == "The trophy didn't fit because the suitcase"
    assert ex.choices[0][1] == ex.choices[1][1] == " was too big."


def test_format_boolq():
    doc = {
        "passage": "Cats are mammals.",
        "question": "are cats mammals",
        "answer": True,
    }
    ex = format_boolq(doc)
    assert ex.answer == 1
    assert ex.choices == [
        ("Cats are mammals.\nQuestion: are cats mammals?\nAnswer:", " no"),
        ("Cats are mammals.\nQuestion: are cats mammals?\nAnswer:", " yes"),
    ]


def test_all_tasks_have_formatters():
    assert set(TASKS) == {
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "winogrande",
        "boolq",
    }


# ---------------------------------------------------------------------------
# encode_choice
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """1 byte = 1 token; special tokens get ids >= 256. Byte-level BPE never
    merges, so expected id sequences can be written down directly."""

    def __init__(self):
        self._special = {tok: 256 + i for i, tok in enumerate(SPECIAL_TOKENS)}

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_single_token(self, token: str) -> int:
        return self._special[token]


def test_encode_choice_base_layout():
    tok = ByteTokenizer()
    ids, cont_start = encode_choice(tok, "ab", "cd")
    assert ids == [tok.encode_single_token(BOS_TOKEN), *b"abcd"]
    assert cont_start == 3  # BOS + "ab"


def test_encode_choice_moves_trailing_context_whitespace():
    tok = ByteTokenizer()
    ids, cont_start = encode_choice(tok, "Answer: ", "yes")
    # " yes" is scored, not "yes": the trailing context space joins the
    # completion so the boundary matches how the text would tokenize as a whole
    assert ids[cont_start:] == list(b" yes")


def test_encode_choice_truncates_left_keeping_completion():
    tok = ByteTokenizer()
    ids, cont_start = encode_choice(tok, "x" * 100, "yz", max_len=10)
    assert len(ids) == 10
    assert ids[0] == tok.encode_single_token(BOS_TOKEN)  # leading BOS survives
    assert ids[cont_start:] == list(b"yz")  # completion survives


def test_encode_choice_chat_renders_user_turn():
    tok = ByteTokenizer()
    ids, cont_start = encode_choice(tok, "Q?", " A", chat=True)
    assert ids[0] == tok.encode_single_token(BOS_TOKEN)
    assert tok.encode_single_token(IM_START) in ids[:cont_start]
    # the leading space is dropped: the assistant header already ends in "\n"
    assert ids[cont_start:] == list(b"A")


# ---------------------------------------------------------------------------
# score_requests
# ---------------------------------------------------------------------------
def _tiny_lm() -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=512,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        window_size=8,
        grad_checkpoint=False,
    ).eval()


def _manual_score(model: TransformerLM, ids: list[int], cont_start: int) -> float:
    with torch.no_grad():
        logprobs = F.log_softmax(model(torch.tensor([ids])).float(), dim=-1)
    return sum(float(logprobs[0, i - 1, ids[i]]) for i in range(cont_start, len(ids)))


def test_score_requests_matches_manual_per_sequence():
    model = _tiny_lm()
    requests = [
        ([7, 3, 9, 4, 2], 2),
        ([5, 1, 8, 8, 3, 2, 9, 4, 6], 4),
        ([9, 9, 1], 1),
    ]
    scores = score_requests(model, requests, pad_id=0, batch_size=2)
    for (ids, cont_start), got in zip(requests, scores):
        assert abs(got - _manual_score(model, ids, cont_start)) < 1e-3


def test_score_requests_padding_invariant():
    # A short row padded up to a longer row's width must score the same as
    # when it is evaluated alone (causal attention ignores right padding).
    model = _tiny_lm()
    short = ([7, 3, 9, 4], 1)
    long = ([5, 1, 8, 8, 3, 2, 9, 4, 6, 2, 7, 1], 2)
    alone = score_requests(model, [short], pad_id=0)[0]
    batched = score_requests(model, [short, long], pad_id=0, batch_size=2)[0]
    assert abs(alone - batched) < 1e-3


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_acc_and_normalization_disagree():
    # The short choice wins on raw sum (fewer tokens to pay for), but the
    # long choice wins per byte -- the exact bias acc_norm corrects.
    examples = [
        MCExample(choices=[("c", "abcdefghij"), ("c", "a")], answer=0),
    ]
    scores = [-2.0, -1.0]  # per byte: -0.2 vs -1.0
    result = summarize(examples, scores)
    assert result["n"] == 1
    assert result["acc"] == 0.0  # raw argmax picks choice 1
    assert result["acc_norm"] == 1.0  # normalized argmax picks choice 0
    assert result["random"] == 0.5


def test_summarize_multiple_examples():
    examples = [
        MCExample(choices=[("c", "x"), ("c", "y")], answer=0),
        MCExample(choices=[("c", "x"), ("c", "y"), ("c", "z"), ("c", "w")], answer=3),
    ]
    scores = [-1.0, -2.0, -5.0, -4.0, -3.0, -2.0]  # ex1 -> 0 (correct), ex2 -> 3
    result = summarize(examples, scores)
    assert result["n"] == 2
    assert result["acc"] == 1.0
    assert result["random"] == (1 / 2 + 1 / 4) / 2
