"""Average several checkpoints into one ("model soup" / checkpoint averaging).

    python scripts/avg_ckpts.py --output weights/base/soup.ckpt \\
        weights/base/picochat-step=140000.ckpt \\
        weights/base/picochat-step=145000.ckpt \\
        weights/base/last.ckpt

Uniformly averaging the weights of several checkpoints from late in one run
(a "tail soup") is a cheap, reliable quality bump: it lands in a flatter
region of the loss basin than any single step, so it generalizes a little
better with no extra training (Model Soups, arXiv:2203.05482). It also
subsumes EMA -- a post-hoc average over saved checkpoints, without the
in-training bookkeeping.

Only parameter tensors are averaged; the result is written as a Lightning
checkpoint that reuses the FIRST input's structure (its `hyper_parameters`,
so the saved model_config -- and thus load_gpt_checkpoint -- keep working)
with the averaged `state_dict` swapped in. All inputs must share the same
architecture (identical state_dict keys and shapes); a mismatch is a hard
error rather than a silent partial average.
"""

import argparse

import torch


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """Uniform mean of matching tensors across `state_dicts`. Floating-point
    tensors are averaged in float64 then cast back to the first's dtype;
    non-float or non-tensor entries (e.g. integer buffers, counters) are taken
    from the first checkpoint unchanged -- averaging them is meaningless. Keys
    and shapes must match across all inputs."""
    ref = state_dicts[0]
    keys = set(ref)
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd) != keys:
            missing = keys.symmetric_difference(sd)
            raise SystemExit(
                f"checkpoint {i} has mismatched keys (e.g. {sorted(missing)[:3]}); "
                "all inputs must share the same architecture"
            )
    out = {}
    for key, ref_val in ref.items():
        if not (torch.is_tensor(ref_val) and torch.is_floating_point(ref_val)):
            out[key] = ref_val
            continue
        acc = torch.zeros_like(ref_val, dtype=torch.float64)
        for i, sd in enumerate(state_dicts):
            val = sd[key]
            if val.shape != ref_val.shape:
                raise SystemExit(
                    f"shape mismatch for {key!r} in checkpoint {i}: "
                    f"{tuple(val.shape)} vs {tuple(ref_val.shape)}"
                )
            acc += val.to(torch.float64)
        out[key] = (acc / len(state_dicts)).to(ref_val.dtype)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="two or more .ckpt files to average")
    p.add_argument("--output", type=str, required=True, help="path for the soup .ckpt")
    args = p.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("need at least two checkpoints to average")

    print(f"loading {len(args.checkpoints)} checkpoints ...", flush=True)
    ckpts = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    for path, ckpt in zip(args.checkpoints, ckpts):
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise SystemExit(f"{path} doesn't look like a Lightning checkpoint")

    averaged = average_state_dicts([c["state_dict"] for c in ckpts])

    # Reuse the first checkpoint's envelope (hyper_parameters carry the
    # model_config load_gpt_checkpoint rebuilds from) with the mean weights.
    soup = dict(ckpts[0])
    soup["state_dict"] = averaged
    # Optimizer/scheduler state would be stale for the averaged weights and is
    # only useful for resuming -- drop it so the soup is inference/finetune
    # only and the file stays small.
    soup.pop("optimizer_states", None)
    soup.pop("lr_schedulers", None)

    torch.save(soup, args.output)
    print(f"averaged {len(args.checkpoints)} checkpoints -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
