#!/usr/bin/env python3
"""Export the neural core of htdemucs to a fixed-window ONNX model.

This is an experimental browser-compatibility probe.  It exports one raw
HTDemucs model, not Demucs' overlap/add, shifts, normalization or audio I/O.
Those deterministic stages remain the responsibility of the caller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import demucs.hdemucs as hdemucs_module
import demucs.htdemucs as htdemucs_module
from demucs.pretrained import get_model


def _export_safe_pad1d(x, paddings, mode="constant", value=0.0):
    """Demucs pad1d without its data-dependent debug assertion."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_right = min(padding_right, extra_pad)
            extra_left = extra_pad - extra_right
            paddings = (padding_left - extra_left, padding_right - extra_right)
            x = F.pad(x, (extra_left, extra_right))
    return F.pad(x, paddings, mode, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/htdemucs_fixed.onnx"))
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    # htdemucs imports pad1d into its own module namespace, so patch both names.
    hdemucs_module.pad1d = _export_safe_pad1d
    htdemucs_module.pad1d = _export_safe_pad1d

    bag = get_model("htdemucs")
    model = bag.models[0].eval()
    samples = round(args.seconds * model.samplerate)
    waveform = torch.randn(1, model.audio_channels, samples) * 0.01
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        reference = model(waveform).cpu().numpy()

    torch.onnx.export(
        model,
        (waveform,),
        args.output,
        input_names=["waveform"],
        output_names=["stems"],
        opset_version=18,
        dynamo=True,
        external_data=True,
    )

    graph = onnx.load(args.output, load_external_data=False)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["stems"], {"waveform": waveform.numpy()})[0]
    error = np.abs(reference - actual)
    print(f"output={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"shape={actual.shape}")
    print(f"max_abs_error={error.max():.8g}")
    print(f"mean_abs_error={error.mean():.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
