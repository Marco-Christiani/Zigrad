//! Zigrad public API.
//!
//! Integration-free tracing produces PR programs. Optional integrations
//!  expose explicit compilation and execution capabilities.
const std = @import("std");

pub const build_options = @import("build_options");

pub const pr = @import("pr");
pub const kernel = @import("kernel.zig");
pub const device = @import("device");
pub const dtype = @import("dtype");
pub const compilation = @import("compilation.zig");
pub const kernelize = @import("kernelize.zig");
pub const outline = @import("outline.zig");
pub const pr_dump = @import("pr_dump.zig");
pub const toolchain = @import("toolchain.zig");
pub const backend = @import("backend.zig");
pub const output = @import("output");
pub const transforms = @import("transforms.zig");
pub const optim = @import("optim.zig");
pub const train = @import("train.zig");
pub const tune = @import("tune.zig");
pub const Cache = @import("cache.zig").Cache;
pub const RuntimeEnv = @import("runtime.zig").RuntimeEnv;
pub const RuntimeLibrary = @import("runtime.zig").RuntimeLibrary;
pub const utils = @import("utils.zig");

/// Optional CUDA driver and runtime compilation capabilities.
pub const cuda = if (build_options.has_nvrtc or build_options.has_cuda_runtime)
    @import("cuda.zig")
else
    struct {};

/// Optional MLIR integration infrastructure.
pub const mlir = if (build_options.has_mlir) @import("mlir.zig") else struct {};

/// Optional StableHLO artifacts used by current compiler integrations.
pub const stablehlo = if (build_options.has_mlir or build_options.has_pjrt or build_options.has_iree)
    @import("stablehlo.zig")
else
    struct {};

/// Optional TVM integration.
pub const tvm = if (build_options.has_tvm) @import("tvm.zig") else struct {};

/// Optional Mirage kernel-provider integration.
pub const mirage = if (build_options.has_mirage) @import("mirage.zig") else struct {};

pub const Tensor = @import("tensor.zig");
/// Mutable queue for composing compiler operations.
pub const Pipeline = compilation.Pipeline;
/// Services supplied while compiler operations run.
pub const CompilationCtx = compilation.Context;
pub const Device = device.Device;
pub const Platform = device.Platform;
pub const Executor = @import("execution.zig");
pub const Backend = backend.Backend;
pub const HostBuffer = utils.HostBuffer;
pub const DType = dtype.DType;
pub const Shape = pr.Shape;
pub const BoundedShape = pr.BoundedShape;

/// Optional PJRT integration.
pub const pjrt = if (build_options.has_pjrt) @import("pjrt.zig") else struct {};

/// Optional IREE integration infrastructure.
pub const iree = if (build_options.has_iree) @import("iree.zig") else struct {};

const trace_mod = @import("trace.zig");
pub const trace = trace_mod.trace;
pub const trace_into = trace_mod.trace_into;
pub const TraceOptions = trace_mod.Options;
const compile_impl = if (build_options.has_zc_cpu) @import("compile.zig") else struct {};

/// Options for tracing the function supplied to `compile`.
pub const CompileOpts = struct {
    /// Trace name and tracing behavior for the authored function.
    trace: TraceOptions = .{},
};

/// Callable produced by the assembled ZC CPU capability. See `zg.compile` for
///  ownership, borrowing, serialization, and result lifetimes.
///  This type is empty when `zc-cpu` is disabled.
pub const CompileCallable = if (build_options.has_zc_cpu) compile_impl.Callable else struct {};

/// Traces a flat Tensor signature and prepares its terminal CPU Product.
///
/// Requires `zc-cpu` and consumer-supplied `zc`, `zcf`, and
///  `zc_implementation_manifest` imports over this module's PR instance.
///  The returned callable owns all retained resources. Its allocator and I/O
///  service are borrowed until `deinit`, and inputs are borrowed during each call.
///  Calls, repeated demands, and teardown require external serialization.
///  Each result owns separate host storage and may outlive the callable while
///  its allocator remains alive. Initialization errors release partial ownership.
///  Unsupported, open, conflicted, unfinished, and callback-failed Runs return
///  an unprepared callable whose `inspect` method exposes the retained evidence.
///
/// The default installs `default_formation`, disables tree and tensor-plan
///  formations, and uses `.reference_required` source association. It requests
///  the baseline x86-64 SysV callable family with default ports and empty donation
///  permissions, matches compatibly, and retains a `catalog_closed` Run.
///  Numerical Contracts use `Profile.exact_scalar` and per-input executable offers.
pub fn compile(comptime func: anytype, allocator: std.mem.Allocator, io: std.Io, specs: anytype, comptime opts: CompileOpts) !*CompileCallable {
    if (comptime !build_options.has_zc_cpu)
        @compileError("zg.compile requires the assembled ZC CPU capability");
    return compile_impl.compile(func, allocator, io, specs, opts.trace);
}

pub const grad = transforms.make_grad;
pub const value_and_grad = transforms.make_value_and_grad;

pub const from_safetensors = utils.safetensors.from_safetensors;
pub const to_safetensors = utils.safetensors.to_safetensors;
pub const FromSafetensorsOpts = utils.safetensors.Opts;
pub const SafetensorsFile = utils.safetensors.SafeTensorsFile;

/// Validates and returns the borrowed PR program.
pub const Validate = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    pub fn run(_: Validate, program: Input, _: *compilation.Context) !Output {
        try pr.validate_program(program);
        return program;
    }
};

test {
    @setEvalBranchQuota(10000);
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(kernel);
    std.testing.refAllDecls(@import("kernel/store.zig"));
    std.testing.refAllDecls(compilation);
    std.testing.refAllDecls(@import("compilation/pipeline.zig"));
    std.testing.refAllDecls(kernelize);
    std.testing.refAllDecls(outline);
    std.testing.refAllDecls(pr_dump);
    _ = @import("tests/higher_order_ad.zig");
    if (build_options.has_zc_cpu) _ = @import("compile.zig");
}
