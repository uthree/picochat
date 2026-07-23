"""Checkpoint averaging (scripts/avg_ckpts.py): the state_dict mean is
correct, non-float entries pass through, and shape/key mismatches error."""

import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "avg_ckpts", Path(__file__).resolve().parents[1] / "scripts" / "avg_ckpts.py"
)
avg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avg)


def test_average_is_the_mean():
    a = {"w": torch.ones(4), "b": torch.zeros(2)}
    b = {"w": torch.ones(4) * 3, "b": torch.ones(2) * 2}
    out = avg.average_state_dicts([a, b])
    assert torch.allclose(out["w"], torch.full((4,), 2.0))
    assert torch.allclose(out["b"], torch.ones(2))


def test_average_three_and_preserves_dtype():
    sds = [{"w": torch.tensor([float(i), float(i)])} for i in (1, 2, 6)]
    out = avg.average_state_dicts(sds)
    assert torch.allclose(out["w"], torch.tensor([3.0, 3.0]))  # (1+2+6)/3
    assert out["w"].dtype == torch.float32
    # bf16 in -> bf16 out (averaged in float64 internally)
    bf = [{"w": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)} for _ in range(2)]
    assert avg.average_state_dicts(bf)["w"].dtype == torch.bfloat16


def test_nonfloat_entries_taken_from_first():
    a = {"step": torch.tensor(10), "buf": torch.tensor([1, 2])}
    b = {"step": torch.tensor(20), "buf": torch.tensor([3, 4])}
    out = avg.average_state_dicts([a, b])
    # integer tensors are not averaged -- first wins (averaging them is meaningless)
    assert out["step"].item() == 10
    assert torch.equal(out["buf"], torch.tensor([1, 2]))


def test_key_mismatch_errors():
    with pytest.raises(SystemExit):
        avg.average_state_dicts([{"w": torch.ones(2)}, {"v": torch.ones(2)}])


def test_shape_mismatch_errors():
    with pytest.raises(SystemExit):
        avg.average_state_dicts([{"w": torch.ones(2)}, {"w": torch.ones(3)}])


def test_end_to_end_soup_file(tmp_path):
    # two minimal Lightning-shaped checkpoints -> a soup that keeps the
    # first's hyper_parameters and drops optimizer state
    paths = []
    for i, scale in enumerate((1.0, 3.0)):
        ckpt = {
            "state_dict": {"model.w": torch.ones(3) * scale},
            "hyper_parameters": {"model_config": {"size": "1b"}},
            "optimizer_states": [{"junk": 1}],
        }
        path = tmp_path / f"c{i}.ckpt"
        torch.save(ckpt, path)
        paths.append(str(path))

    out = tmp_path / "soup.ckpt"
    import sys

    argv = sys.argv
    sys.argv = ["avg_ckpts.py", *paths, "--output", str(out)]
    try:
        avg.main()
    finally:
        sys.argv = argv

    soup = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.allclose(soup["state_dict"]["model.w"], torch.full((3,), 2.0))
    assert soup["hyper_parameters"]["model_config"]["size"] == "1b"
    assert "optimizer_states" not in soup  # dropped: stale for averaged weights
