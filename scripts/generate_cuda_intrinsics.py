"""Serialize TVM's CUDA tensor intrinsics for native registration."""

import argparse
import importlib
import json
from pathlib import Path

from tvm.tir import PrimFunc, TensorIntrin

import tvm


def registered_cuda_intrinsics() -> dict[str, TensorIntrin]:
    """Replay TVM's CUDA module and capture each registered intrinsic."""
    registrations: dict[str, TensorIntrin] = {}
    register = TensorIntrin.register
    enabled = tvm.runtime.enabled

    def capture(
        name: str,
        desc: PrimFunc,
        impl: PrimFunc,
        override: bool = False,
    ) -> TensorIntrin:
        if name in registrations and not override:
            raise RuntimeError(f"duplicate TensorIntrin registration: {name}")
        intrinsic = TensorIntrin(desc, impl)
        registrations[name] = intrinsic
        return register(name, desc, impl, override=True)

    TensorIntrin.register = staticmethod(capture)
    # The package initializer always imports CUDA, then conditionally imports
    # target-specific CPU modules using this feature probe.
    tvm.runtime.enabled = lambda _: False
    try:
        importlib.import_module("tvm.tir.tensor_intrin.cuda")
    finally:
        TensorIntrin.register = staticmethod(register)
        tvm.runtime.enabled = enabled

    return registrations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    entries = []
    for name, intrinsic in sorted(registered_cuda_intrinsics().items()):
        entries.extend((name, intrinsic))
    if not entries:
        raise RuntimeError("TVM registered no CUDA tensor intrinsics")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = tvm.ir.save_json(tvm.runtime.convert(entries))
    args.output.write_text(json.dumps(json.loads(serialized), separators=(",", ":")))
    print(f"serialized {len(entries) // 2} CUDA tensor intrinsics to {args.output}")


if __name__ == "__main__":
    main()
