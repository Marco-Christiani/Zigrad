const std = @import("std");
const zg = @import("zigrad");
const build_options = zg.build_options;
const demos = @import("demos.zig");
const iree_compile_cmd = @import("iree/compile_cmd.zig");
const llm_demo = @import("llm_demo.zig");
const pjrt_aot = @import("demos/pjrt_aot.zig");
const cli = @import("cli.zig");
const log = std.log.scoped(.@"zg/main");

pub fn main(init: std.process.Init) !void {
    const env = zg.RuntimeEnv.from_init(init);
    const gpa = env.allocator;

    var parsed = cli.parse(env, init.minimal.args) catch |err| {
        if (err == error.HelpShown) return;
        if (err == error.InvalidArguments) std.process.exit(2);
        return err;
    };
    defer parsed.deinit();

    if (comptime build_options.has_tvm) {
        try zg.tvm.runtime.configure(.from_environ(
            env.environ,
            tvm_surface_for_command(parsed.invocation.command),
        ));
    }

    const global = parsed.invocation.global;
    const outputs = OutputOptions{
        .pr = global.dump_pr,
        .mlir = global.dump_mlir,
        .optimized_hlo = global.dump_optimized_hlo,
        .kernels = global.dump_kernels,
    };
    try validate_output_options(parsed.invocation.command, outputs);

    return switch (parsed.invocation.command) {
        .pr => |command| dispatch_pr(env, gpa, command),
        .tvm => |command| dispatch_tvm(env, gpa, command),
        .iree => |command| dispatch_iree(env, gpa, command, outputs),
        .pjrt => |command| dispatch_pjrt(env, gpa, .{ .pjrt = command }, outputs, global.quiet),
        .demo => |command| dispatch_demo(env, gpa, command, outputs, global.quiet),
    };
}

fn tvm_surface_for_command(command: cli.Command) zg.tvm.runtime.Surface {
    return switch (command) {
        .tvm => |tvm_command| switch (tvm_command) {
            .symbols => |opts| if (opts.load_compiler) .compiler else .ffi,
            .check_load, .tune => .compiler,
            .run => .runtime,
            .render_matmul, .render_attention => .ffi,
        },
        .demo => |demo_command| switch (demo_command) {
            .kernel_provider => .compiler,
            else => .ffi,
        },
        else => .ffi,
    };
}

const OutputOptions = struct {
    pr: ?zg.pr.dump.Config,
    mlir: ?zg.output.Config,
    optimized_hlo: ?zg.output.Config,
    kernels: ?zg.output.Config,
};

const OutputSupport = struct {
    pr: bool = false,
    mlir: bool = false,
    optimized_hlo: bool = false,
    kernels: bool = false,
};

const OutputValidationError = error{UnsupportedOutput};

fn validate_output_options(
    command: cli.Command,
    outputs: OutputOptions,
) OutputValidationError!void {
    const support: OutputSupport = switch (command) {
        .demo => |demo| .{
            .pr = true,
            .mlir = true,
            .optimized_hlo = demo_backend(demo) == .pjrt,
            .kernels = switch (demo) {
                .kernel_provider => true,
                else => false,
            },
        },
        .iree => .{ .pr = true, .mlir = true },
        else => .{},
    };
    if (outputs.pr != null and !support.pr)
        return unsupported_output("--dump-pr");
    if (outputs.mlir != null and !support.mlir)
        return unsupported_output("--dump-mlir");
    if (outputs.optimized_hlo != null and !support.optimized_hlo)
        return unsupported_output("--dump-optimized-hlo");
    if (outputs.kernels != null and !support.kernels)
        return unsupported_output("--dump-kernels");
}

fn unsupported_output(option: []const u8) OutputValidationError {
    log.err("{s} is not available for the selected command and backend", .{option});
    return error.UnsupportedOutput;
}

fn dispatch_pr(env: zg.RuntimeEnv, gpa: std.mem.Allocator, command: cli.PrCommand) !void {
    return switch (command) {
        .print_demo => demos.print_pr(env.io, gpa, env.environ),
        .render => |opts| zg.pr.tool.run(
            env.io,
            gpa,
            opts.path,
            .{ .render = opts.format },
        ),
        .info => |opts| zg.pr.tool.run(env.io, gpa, opts.path, .info),
    };
}

fn dispatch_tvm(env: zg.RuntimeEnv, gpa: std.mem.Allocator, command: cli.TvmCommand) !void {
    switch (command) {
        .render_matmul => |opts| {
            const palette = try parse_palette(opts.palette);
            return try demos.print_tvm_kernelize_pr(
                env.io,
                gpa,
                env.environ,
                opts.sweep_palettes,
                palette,
            );
        },
        .render_attention => |opts| {
            const palette = try parse_palette(opts.palette);
            return try demos.print_tvm_attention_pr(
                env.io,
                gpa,
                env.environ,
                opts.sweep_palettes,
                palette,
            );
        },
        .symbols => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            return try demos.dump_tvm_ffi_symbols(env.io, gpa, opts.load_compiler);
        },
        .check_load => {
            if (comptime !build_options.has_tvm) return try require_tvm();
            try zg.tvm.runtime.ensure_loaded(.compiler);
            log.info("tvm compiler load check passed", .{});
        },
        .tune => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            const shape = try parse_shape(opts.shape orelse "128x128x128");
            const target_kind: zg.tvm.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
            const selected_device = zg.Device{
                .platform = if (target_kind == .cuda) .cuda else .cpu,
            };
            const compile_config = try zg.tvm.CompileConfig.from_environ(
                env.environ,
                target_kind,
            );
            const cache = try zg.Cache.init(env.io, env.environ, .{});
            _ = try zg.tvm.tune_matmul(
                env.io,
                gpa,
                &cache,
                .{
                    .m = @intCast(shape.m),
                    .n = @intCast(shape.n),
                    .k = @intCast(shape.k),
                },
                .{
                    .compile = compile_config,
                    .device = selected_device,
                    .max_trials = opts.trials orelse 64,
                    .trials_per_iter = opts.trials_per_iter orelse 16,
                },
            );
        },
        .run => |opts| {
            if (comptime !build_options.has_tvm) return try require_tvm();
            const shape = try parse_shape(opts.shape orelse "128x128x128");
            const target_kind: zg.tvm.TargetKind = if (opts.cuda or opts.gpu) .cuda else .cpu;
            const cache = try zg.Cache.init(env.io, env.environ, .{});
            const build_identity = try zg.tvm.CompileConfig.resolve_build_identity(
                env.environ,
            );
            return try run_tvm_demo(
                env.io,
                gpa,
                @intCast(shape.m),
                @intCast(shape.n),
                @intCast(shape.k),
                target_kind,
                build_identity,
                &cache,
            );
        },
    }
}

fn dispatch_iree(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    command: cli.IreeCommand,
    outputs: OutputOptions,
) !void {
    if (comptime build_options.has_iree and build_options.has_mlir) {
        return switch (command) {
            .compile => |opts| iree_compile_cmd.run(
                env.io,
                gpa,
                env.environ,
                .{
                    .input = opts.path,
                    .output = opts.output,
                    .target = opts.target,
                    .entry = opts.entry,
                    .compiler_arguments = opts.compiler_arguments,
                    .pr = outputs.pr,
                    .mlir = outputs.mlir,
                },
            ),
        };
    }
    log.err("IREE commands require the opt-in IREE and MLIR integrations", .{});
    return error.IreeBackendDisabled;
}

fn dispatch_demo(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    command: cli.DemoCommand,
    outputs: OutputOptions,
    quiet: bool,
) !void {
    return switch (demo_backend(command)) {
        .pjrt => dispatch_pjrt(env, gpa, .{ .demo = command }, outputs, quiet),
        .iree => dispatch_iree_demo(env, gpa, command, outputs, quiet),
    };
}

fn demo_backend(command: cli.DemoCommand) cli.DemoBackend {
    return switch (command) {
        .basic => |opts| opts.backend,
        .custom_call_negative => |opts| opts.backend,
        .kernel_provider => .pjrt,
        .vjp => |opts| opts.backend,
        .train => |opts| opts.backend,
        .llm_train => |opts| opts.backend,
    };
}

const PjrtWork = union(enum) {
    pjrt: cli.PjrtCommand,
    demo: cli.DemoCommand,
};

fn dispatch_pjrt(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    work: PjrtWork,
    outputs: OutputOptions,
    quiet: bool,
) !void {
    if (comptime build_options.has_pjrt) {
        const plugin_path = env.environ.get("PJRT_PLUGIN_PATH") orelse {
            log.err("PJRT_PLUGIN_PATH is not set", .{});
            return error.PjrtPluginPathNotSet;
        };

        const pjrt_options = zg.pjrt.config.from_environ(env.environ) catch |err| {
            log.err("invalid PJRT runtime configuration: {s}", .{@errorName(err)});
            return err;
        };
        var pjrt_client = try zg.pjrt.Client.init(gpa, plugin_path, pjrt_options);
        defer pjrt_client.deinit();

        const devices = try pjrt_client.get_devices(gpa);
        defer gpa.free(devices);
        if (devices.len == 0) {
            log.err("no devices available from PJRT", .{});
            return error.NoDevices;
        }
        const device = devices[0];
        var execution = try zg.pjrt.Execution.init(&pjrt_client, device, .{});
        var ctx = zg.CompilationCtx{
            .allocator = gpa,
            .io = env.io,
            .device = execution.interface.device,
        };
        var backend = zg.pjrt.Backend.init(&execution, .{});
        return switch (work) {
            .pjrt => |command| dispatch_pjrt_artifact(
                env,
                command,
                &ctx,
                &pjrt_client,
                &execution,
                &backend,
            ),
            .demo => |command| dispatch_pjrt_demo(
                env,
                command,
                &ctx,
                &pjrt_client,
                &execution,
                &backend,
                outputs,
                quiet,
            ),
        };
    }

    log.err("this command requires the opt-in PJRT integration", .{});
    return error.PjrtDisabled;
}

fn dispatch_pjrt_artifact(
    env: zg.RuntimeEnv,
    command: cli.PjrtCommand,
    ctx: *zg.CompilationCtx,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
) !void {
    const gpa = ctx.allocator;
    return switch (command) {
        .aot_demo => {
            if (comptime !build_options.has_mlir) {
                log.err("pjrt aot-demo requires the opt-in MLIR integration", .{});
                return error.MlirDisabled;
            }
            return try pjrt_aot.run(
                ctx,
                client,
                execution,
                backend,
            );
        },
        .cache => |cache| switch (cache) {
            .save => |opts| {
                if (comptime !build_options.has_mlir) {
                    log.err("pjrt cache save requires the opt-in MLIR integration", .{});
                    return error.MlirDisabled;
                }

                var program = zg.pr.Program.init(gpa);
                defer program.deinit();
                const entry = try demos.build_demo_program(&program);
                try program.set_entry(entry);

                var pipeline = try zg.pjrt.pipeline.create(gpa, backend, .{});
                defer pipeline.deinit();
                var loaded_program = try pipeline.run(
                    zg.Executor.LoadedProgram,
                    &program,
                    ctx,
                );
                defer loaded_program.deinit();
                const serialized = try (try execution.loaded(loaded_program)).serialize(
                    client.api,
                    gpa,
                );
                defer gpa.free(serialized);

                try demos.write_bytes_to_path(env.io, opts.path, serialized);
                log.info(
                    "wrote PJRT executable artifact: {d} bytes -> {s}",
                    .{ serialized.len, opts.path },
                );
            },
            .run => |opts| {
                const serialized = try demos.read_bytes_from_path(env.io, gpa, opts.path);
                defer gpa.free(serialized);

                var artifact = zg.pjrt.Artifact{
                    .client = client,
                    .loaded = try client.load_serialized_executable(serialized, null),
                };
                errdefer artifact.deinit();
                var loaded_program = try backend.loader.interface.load(&artifact);
                defer loaded_program.deinit();
                return try demos.run_demo_executable(gpa, loaded_program);
            },
        },
    };
}

fn dispatch_pjrt_demo(
    env: zg.RuntimeEnv,
    command: cli.DemoCommand,
    ctx: *zg.CompilationCtx,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
    outputs: OutputOptions,
    quiet: bool,
) !void {
    if (comptime !build_options.has_mlir) {
        log.err("the current executable demos require the opt-in MLIR integration", .{});
        return error.MlirDisabled;
    }

    switch (command) {
        .kernel_provider => |opts| {
            const providers = try parse_provider_kinds(opts.provider orelse "tvm");
            const shape = try parse_shape(opts.shape orelse "128x128x128");
            const dtype = std.meta.stringToEnum(zg.pr.DType, opts.dtype orelse "f32") orelse
                return error.InvalidArgument;
            if (dtype != .f16 and dtype != .bf16 and dtype != .f32)
                return error.InvalidArgument;
            const pattern = std.meta.stringToEnum(
                demos.KernelProviderDemoPattern,
                opts.pattern orelse "epilogue",
            ) orelse return error.InvalidArgument;
            return try demos.run_kernel_provider_demo(
                ctx,
                client,
                execution,
                backend,
                env.environ,
                .{
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .dtype = dtype,
                    .pattern = pattern,
                    .entry_label = "main",
                    .pr = outputs.pr,
                    .mlir = outputs.mlir,
                    .optimized_hlo = outputs.optimized_hlo,
                    .kernels = if (outputs.kernels) |config| config.target else null,
                },
                providers.slice(),
            );
        },
        else => {},
    }

    var pipeline = try create_pjrt_demo_pipeline(ctx.allocator, backend, execution, outputs);
    defer pipeline.deinit();
    return try run_portable_demo(env, command, ctx, &pipeline, quiet);
}

fn run_portable_demo(
    env: zg.RuntimeEnv,
    command: cli.DemoCommand,
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
    quiet: bool,
) !void {
    return switch (command) {
        .basic => run_basic_demo(ctx, pipeline),
        .custom_call_negative => demos.run_custom_call_negative(
            ctx,
            pipeline,
        ),
        .kernel_provider => unreachable,
        .vjp => demos.run_vjp_demo(ctx, pipeline),
        .train => |opts| demos.run_train_demo(
            ctx,
            pipeline,
            opts.warmup orelse 0,
            opts.steps orelse 8,
            quiet,
        ),
        .llm_train => |opts| llm_demo.run_llm_train_demo(
            ctx,
            pipeline,
            env.environ,
            opts.warmup orelse 0,
            opts.steps orelse 8,
            quiet,
        ),
    };
}

fn create_pjrt_demo_pipeline(
    allocator: std.mem.Allocator,
    backend: *zg.pjrt.Backend,
    execution: *zg.pjrt.Execution,
    outputs: OutputOptions,
) !zg.Pipeline {
    var pipeline = zg.Pipeline.init(allocator);
    errdefer pipeline.deinit();
    try add_mlir_input(&pipeline, outputs);
    try pipeline.add(&backend.interface);
    if (outputs.optimized_hlo) |config| {
        try pipeline.add(zg.pjrt.DumpOptimizedHlo{
            .execution = execution,
            .config = config,
        });
    }
    return pipeline;
}

fn create_iree_demo_pipeline(
    allocator: std.mem.Allocator,
    backend: *zg.iree.Backend,
    outputs: OutputOptions,
) !zg.Pipeline {
    var pipeline = zg.Pipeline.init(allocator);
    errdefer pipeline.deinit();
    try add_mlir_input(&pipeline, outputs);
    try pipeline.add(&backend.interface);
    return pipeline;
}

fn add_mlir_input(
    pipeline: *zg.Pipeline,
    outputs: OutputOptions,
) !void {
    try pipeline.add(zg.pr.Validate{});
    if (outputs.pr) |config| {
        try pipeline.add(zg.pr.dump.Dump{ .config = config });
    }
    try pipeline.add(zg.pr.transform.outline.Pass{});
    try pipeline.add(zg.mlir.stablehlo.Lower{
        .config = .{
            .encoding = if (outputs.mlir == null) .binary else .text,
        },
    });
    if (outputs.mlir) |config| {
        try pipeline.add(zg.stablehlo.Dump{ .config = config });
    }
}

fn parse_palette(name: ?[]const u8) !?zg.pr.zxpr.style.Palette {
    const value = name orelse return null;
    return std.meta.stringToEnum(zg.pr.zxpr.style.Palette, value) orelse
        error.InvalidPalette;
}

fn require_tvm() error{TvmUnavailable}!void {
    log.err("this command requires the opt-in TVM integration", .{});
    return error.TvmUnavailable;
}

fn run_tvm_demo(
    io: std.Io,
    gpa: std.mem.Allocator,
    m: i64,
    n: i64,
    k: i64,
    target_kind: if (build_options.has_tvm) zg.tvm.TargetKind else void,
    build_identity: if (build_options.has_tvm) zg.tvm.BuildIdentity else void,
    artifact_cache: *const zg.Cache,
) !void {
    const shape: zg.tvm.MatmulShape = .{ .m = m, .n = n, .k = k };
    const selected_device = zg.Device{
        .platform = if (target_kind == .cuda) .cuda else .cpu,
    };
    const cached = try zg.tvm.CachedMatmul.load(
        io,
        gpa,
        artifact_cache,
        shape,
        build_identity,
        target_kind,
        selected_device,
    ) orelse return error.NoTuningRecords;
    defer cached.deinit();

    const a = try gpa.alloc(f32, @intCast(m * k));
    defer gpa.free(a);
    const b = try gpa.alloc(f32, @intCast(k * n));
    defer gpa.free(b);
    const result = try gpa.alloc(f32, @intCast(m * n));
    defer gpa.free(result);

    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1;
    @memset(result, 0);

    const start = std.Io.Timestamp.now(io, .awake);
    try cached.execute(.f32, a, b, result);

    const elapsed = start.untilNow(io, .awake);
    const ms = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1_000_000.0;
    std.log.info("TVM matmul {d}x{d}x{d} ({s}): {d:.3} ms", .{ m, n, k, @tagName(target_kind), ms });

    var expected: f32 = 0;
    for (0..@intCast(k)) |inner| {
        const a_val: f32 = @as(f32, @floatFromInt((@as(usize, @intCast(m - 1)) * @as(usize, @intCast(k)) + inner) % 7)) * 0.1;
        const b_val: f32 = @as(f32, @floatFromInt((inner * @as(usize, @intCast(n)) + @as(usize, @intCast(n - 1))) % 11)) * 0.1;
        expected += a_val * b_val;
    }
    std.log.info("result[m-1,n-1] = {d:.6} (expected ~{d:.6})", .{ result[@as(usize, @intCast(m * n - 1))], expected });
}

// TODO(cli): Replace this fixed buffer when provider registration becomes dynamic.
const ProviderKindList = struct {
    buf: [2]demos.KernelProviderDemoKind,
    len: usize,

    fn slice(self: *const ProviderKindList) []const demos.KernelProviderDemoKind {
        return self.buf[0..self.len];
    }
};

fn parse_provider_kinds(s: []const u8) !ProviderKindList {
    var result: ProviderKindList = .{ .buf = undefined, .len = 0 };
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " ");
        if (result.len >= result.buf.len) return error.TooManyProviders;
        result.buf[result.len] = std.meta.stringToEnum(demos.KernelProviderDemoKind, trimmed) orelse return error.InvalidArgument;
        result.len += 1;
    }
    if (result.len == 0) return error.InvalidArgument;
    return result;
}

fn dispatch_iree_demo(
    env: zg.RuntimeEnv,
    gpa: std.mem.Allocator,
    command: cli.DemoCommand,
    outputs: OutputOptions,
    quiet: bool,
) !void {
    if (comptime build_options.has_iree and build_options.has_mlir) {
        const config = zg.iree.Config.from_environ(env.environ);
        var runtime = try zg.iree.Runtime.init(gpa, .registered, config.runtime);
        defer runtime.deinit();
        var execution = zg.iree.Execution.init(gpa, &runtime, config.runtime);

        var ctx = zg.CompilationCtx{
            .allocator = gpa,
            .io = env.io,
            .device = execution.interface.device,
        };
        var backend = zg.iree.Backend.init(&execution, config.compiler, "module.main");
        var pipeline = try create_iree_demo_pipeline(gpa, &backend, outputs);
        defer pipeline.deinit();
        return try run_portable_demo(
            env,
            command,
            &ctx,
            &pipeline,
            quiet,
        );
    }

    log.err("the IREE backend requires the opt-in IREE and MLIR integrations", .{});
    return error.IreeBackendDisabled;
}

fn run_basic_demo(
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
) !void {
    var program = zg.pr.Program.init(ctx.allocator);
    defer program.deinit();
    const function = try demos.build_demo_program(&program);
    try program.set_entry(function);
    var loaded_program = try pipeline.run(zg.Executor.LoadedProgram, &program, ctx);
    defer loaded_program.deinit();
    return try demos.run_demo_executable(
        ctx.allocator,
        loaded_program,
    );
}

const MatmulShape = struct { m: i64, n: i64, k: i64 };

fn parse_shape(s: []const u8) !MatmulShape {
    var parts = std.mem.splitScalar(u8, s, 'x');
    const m = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const n = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    const k = try std.fmt.parseInt(i64, parts.next() orelse return error.InvalidShape, 10);
    if (parts.next() != null or m <= 0 or n <= 0 or k <= 0) return error.InvalidShape;
    return .{ .m = m, .n = n, .k = k };
}
