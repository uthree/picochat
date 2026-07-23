"""The in-repo SFT seed sets (identity / safety) parse, are well-formed
ChatML conversations, and load through the local-JSONL conversation source."""

import json
from pathlib import Path

import pytest

from picochat.data.sources import (
    ChatDatasetSpec,
    _is_local_jsonl,
    iter_conversations,
)

SEED_DIR = Path(__file__).resolve().parents[1] / "data_seeds"
SEED_FILES = ["identity.jsonl", "safety.jsonl"]


@pytest.mark.parametrize("name", SEED_FILES)
def test_seed_file_is_wellformed(name):
    path = SEED_DIR / name
    assert path.exists(), f"missing seed file {name}"
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)  # valid JSON
            messages = rec["messages"]
            assert len(messages) >= 2
            for m in messages:
                assert m["role"] in ("system", "user", "assistant")
                assert isinstance(m["content"], str) and m["content"].strip()
            # a conversation must end on an assistant turn (the trained target)
            assert messages[-1]["role"] == "assistant"
            # exactly one system turn at most, and only at the front
            roles = [m["role"] for m in messages]
            assert roles.count("system") <= 1
            if "system" in roles:
                assert roles[0] == "system"
            n += 1
    assert n >= 5  # each seed set is small but non-trivial


@pytest.mark.parametrize("name", SEED_FILES)
def test_seed_file_loads_through_local_source(name):
    spec = ChatDatasetSpec(path=f"jsonl:{SEED_DIR / name}")
    assert _is_local_jsonl(spec) is not None
    convos = list(iter_conversations(spec, limit=3))
    assert 1 <= len(convos) <= 3
    assert all(isinstance(c, list) and c for c in convos)


def test_is_local_jsonl_detects_forms(tmp_path):
    # "jsonl:" prefix form (need not exist yet)
    assert _is_local_jsonl(ChatDatasetSpec(path="jsonl:whatever.jsonl"))
    # bare .jsonl path only when the file exists
    p = tmp_path / "x.jsonl"
    assert _is_local_jsonl(ChatDatasetSpec(path=str(p))) is None
    p.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert _is_local_jsonl(ChatDatasetSpec(path=str(p))) == str(p)
    # a Hub dataset id is not local
    assert _is_local_jsonl(ChatDatasetSpec(path="HuggingFaceTB/smoltalk")) is None


def test_local_jsonl_accepts_bare_list(tmp_path):
    p = tmp_path / "bare.jsonl"
    p.write_text(
        '[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]\n'
    )
    convos = list(iter_conversations(ChatDatasetSpec(path=f"jsonl:{p}")))
    assert convos == [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    ]
