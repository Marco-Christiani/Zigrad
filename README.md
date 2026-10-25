<p align="center">
  <img src="./assets/zg-logo.svg" width=350>
</p>

<p align="center">
 <img src="https://img.shields.io/github/license/Marco-Christiani/zigrad?style=flat&logo=opensourceinitiative" alt="license">
 <img src="https://img.shields.io/github/last-commit/Marco-Christiani/zigrad?style=flat&logo=git&logoColor=white" alt="last-commit">
 <img src="https://img.shields.io/github/languages/top/Marco-Christiani/zigrad?style=flat&color=F7A41D" alt="repo-top-language">
 <img alt="Discord" src="https://img.shields.io/discord/1325584101809324113?style=flat">
</p>
<br>
<p align="center"><strong><i>Supporting AI innovation from ideation to results.</i></strong></p>

---

> **Zigrad is under active development.**
> This is the `modular` rewrite - a ground-up redesign as a unified multi-level compiler and ML framework. APIs and architecture are evolving rapidly.

---

Zigrad is a unified machine-learning stack for carrying one authored computation across the AI lifecycle without translation into parallel codebases.

This tree contains an ML framework built on several external integrations while Zigrad's compiler stack is in progress. The MLIR, XLA, IREE, TVM, and Mirage integrations remain usable during that work.

<p align="center">
  <img src="./assets/zigrad-pipeline.svg" width=700>
</p>

## Capabilities

- **Forward and reverse-mode AD** - JVP and VJP transformations over PR
- **Core compiler passes** - outlining, delegation (eg to TVM, Mirage), compile-time AD rewrites, etc.
- **OpenXLA backend** - Lower to the [StableHLO](https://github.com/openxla/stablehlo) dialect and execute via PJRT CPU/CUDA
- **IREE backend** - AOT compilation + minimal runner for embedded targets
- **Kernel delegation** - [TVM](https://tvm.apache.org/) and the Zigrad C API for [Mirage](https://github.com/mirage-project/mirage)
<!-- - **Nix hermetic infrastructure** - hermetic devshell with LLVM, XLA, PJRT, TVM, and StableHLO pinned -->

## Consumption Surfaces

The repository defines three build-facing surfaces:

1. The Zig source package exports the `zigrad` module, which supports integration-free tracing. To assemble `zg.compile`, request `zc-cpu=true` from the dependency and add the `zc`, `zcf`, and `zc_implementation_manifest` imports to its `zigrad` module. Build those imports over the dependency's `pr` module. A program calls `zg.compile(func, allocator, io, specs, .{})` and then `callable.call(.{...})`. A call with the capability disabled reports `zg.compile requires the assembled ZC CPU capability` at compile time. ZCF's `examples/zg/compile_cpu` is the maintained consumer.
2. Nix packages install the `zigrad` CLI. Named configurations (see below) build that CLI with a set of integrations and provide the runtime environment needed to execute it.
3. Devshells expose a broad composition of external compile and runtime inputs for iteration purposes.

## Getting Started

> **Not user friendly yet**
> Actively working on a proper on-ramp for users, this is not a primary concern at the moment.
> The external integrations are expensive to build and are not yet a polished installation.

Nix is the standard build and execution interface. The core package is the default:

```sh
nix build .#zigrad
nix run .#zigrad -- pr print-demo
nix flake check
```

Named configurations describe valid combinations of integrations. Each demands its transitive build and runtime dependencies (e.g., XLA + IREE demands MLIR + StableHLO dialect). Every configuration depends on Zigrad core.

| Configuration | Included integrations |
| --- | --- |
| `zigrad` | Core only |
| `xla:cpu` | XLA backend |
| `xla:cuda` | XLA CUDA backend |
| `iree:cpu` | IREE CPU backend |
| `iree:cuda` | IREE CUDA backend |
| `xla:cpu+iree:cpu` | XLA CPU and IREE CPU backends |
| `xla:cuda+iree:cuda` | XLA CUDA + IREE CUDA backends |
| `xla:cuda+iree:cpu` | XLA CUDA + IREE CPU backends |
| `tvm:cpu` | TVM CPU tuning and execution |
| `tvm:cuda` | TVM CUDA tuning and execution |
| `xla:cpu+tvm:cpu` | XLA CPU backend + TVM CPU kernel specialization |
| `xla:cuda+tvm:cuda` | XLA CUDA backend + TVM kernel specialization |
| `xla:cuda+mirage:cuda` | XLA CUDA backend + Mirage kernel specialization |
| `xla:cuda+iree:cpu+tvm:cuda+mirage:cuda` | Broad devshell composition |

TVM, Mirage, etc derive the runtime NVRTC target from the selected device. However,
what architectures the upstream packages build for can be restricted by setting
`cudaArchitectures` in `local-build-cfg.nix`. This can significantly reduce the
cost/time to build external packages since these builds involve compiling many kernels.
**This requires passing `--impure` to nix commands so it can read the local file, if this flag
is omitted the config file is ignored. Simply add `--impure` to any of the documented nix commands.**

For example:

```sh
nix run '.#"xla:cpu"' -- demo vjp
nix run '.#"iree:cpu"' -- demo basic --backend=iree
nix run '.#"tvm:cpu"' -- tvm check-load
```

Devshell:

```sh
nix develop # or direnv allow
zigrad --help
```

The default devshell is intentionally broad (it is expensive to build) and exports the composed external input roots and the matching `ZG_ZIG_BUILD_ARGS` for direct Zig iteration.
Zigrad remains buildable without Nix *when the external dependencies are available:*

```sh
zig build
zig build test --summary all
```

## Contributing

- [Join the Discord](https://discord.gg/JWSSfWj3Uf) and head to the dev channels
- Open an issue before starting a PR
