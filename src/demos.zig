const std = @import("std");
const zg = @import("zigrad");

const Tensor = zg.Tensor;
const log = std.log.scoped(.@"zg/demos");

/// Demo program: out = (mm(A, B) + C) * C where A: 2x3, B: 3x2, C: 2x2.
pub fn build_demo_program(program: *zg.pr.Program) !zg.pr.FunctionId {
    var b = try zg.pr.FunctionBuilder.init(program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 3 });
    const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    const mm_id = try b.mm(a_id, b_id);
    const add_id = try b.add(mm_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(.{ .returns = &.{out_id} });
    return try program.add_function(func);
}

pub fn write_bytes_to_path(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}

pub fn read_bytes_from_path(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

pub fn run_demo_executable(
    allocator: std.mem.Allocator,
    loaded_program: zg.Executor.LoadedProgram,
) !void {
    const executor = loaded_program.executor;
    const A = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const B = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    };
    const C = [_]f32{
        2.0, 2.0,
        2.0, 2.0,
    };

    // TODO(api): Decide whether tensor initialization infers byte conversion.
    var host_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .borrow = std.mem.sliceAsBytes(&A) });
    defer host_a.deinit();
    var host_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .borrow = std.mem.sliceAsBytes(&B) });
    defer host_b.deinit();
    var host_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&C) });
    defer host_c.deinit();

    const dev_a = try executor.upload(host_a.host_data(), .f32, host_a.dims());
    defer executor.release(dev_a);
    const dev_b = try executor.upload(host_b.host_data(), .f32, host_b.dims());
    defer executor.release(dev_b);
    const dev_c = try executor.upload(host_c.host_data(), .f32, host_c.dims());
    defer executor.release(dev_c);

    var outputs: [1]zg.Executor.Buffer = undefined;
    const event = try executor.invoke(
        loaded_program,
        &.{ dev_a, dev_b, dev_c },
        &outputs,
        .{},
    );
    defer if (event) |completion| executor.release_event(completion);
    defer executor.release(outputs[0]);

    var out_host = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = allocator });
    defer out_host.deinit();
    if (try executor.download(outputs[0], out_host.host_data_mut())) |completion| {
        defer executor.release_event(completion);
        try executor.wait(completion);
    }

    const out = out_host.as_const_slice(f32);
    std.debug.assert(out.len == 4);
    const expected = [_]f32{
        120.0, 132.0,
        282.0, 312.0,
    };

    for (out, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > 1e-4) {
            log.err("mismatch[{d}]: got {d}, expected {d}", .{ i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }

    log.info("OK: demo output matches expected", .{});
}

pub fn run_custom_call_negative(
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
) !void {
    const allocator = ctx.allocator;

    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var b = try zg.pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const outputs = (try b.custom_call(.{
        .target_name = "zigrad.test.missing_handler",
        .has_side_effect = false,
    }, &.{x}, &.{x.aval})).outputs;
    const y = outputs[0];
    const func = try b.finish(.{ .returns = &.{y} });
    const function = try program.add_function(func);
    try program.set_entry(function);

    var exe = pipeline.run(zg.Executor.LoadedProgram, &program, ctx) catch |err| {
        log.info("OK: custom_call compile failed as expected: {s}", .{@errorName(err)});
        return;
    };
    defer exe.deinit();

    log.err("unexpected: custom_call compiled without a handler", .{});
    return error.UnexpectedSuccess;
}

pub fn run_vjp_demo(
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
) !void {
    const allocator = ctx.allocator;

    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    const forward = try build_demo_program(&program);

    const derived = try zg.pr.ad.vjp(allocator, &program, forward, "main_vjp", .{});
    try program.set_entry(derived);

    var exe = try pipeline.run(zg.Executor.LoadedProgram, &program, ctx);
    defer exe.deinit();
    const executor = exe.executor;

    const A = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const B = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    };
    const C = [_]f32{
        2.0, 2.0,
        2.0, 2.0,
    };
    // The final parameter seeds the 2x2 output cotangent.
    const CtOut = [_]f32{
        1.0, 1.0,
        1.0, 1.0,
    };

    var host_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .borrow = std.mem.sliceAsBytes(&A) });
    defer host_a.deinit();
    var host_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .borrow = std.mem.sliceAsBytes(&B) });
    defer host_b.deinit();
    var host_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&C) });
    defer host_c.deinit();
    var host_ct = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&CtOut) });
    defer host_ct.deinit();

    const dev_a = try executor.upload(host_a.host_data(), .f32, host_a.dims());
    defer executor.release(dev_a);
    const dev_b = try executor.upload(host_b.host_data(), .f32, host_b.dims());
    defer executor.release(dev_b);
    const dev_c = try executor.upload(host_c.host_data(), .f32, host_c.dims());
    defer executor.release(dev_c);
    const dev_ct = try executor.upload(host_ct.host_data(), .f32, host_ct.dims());
    defer executor.release(dev_ct);

    var outputs: [3]zg.Executor.Buffer = undefined;
    const event = try executor.invoke(
        exe,
        &.{ dev_a, dev_b, dev_c, dev_ct },
        &outputs,
        .{},
    );
    defer if (event) |completion| executor.release_event(completion);
    defer for (outputs) |output| executor.release(output);

    var out_a = try Tensor.host(.f32, &.{ 2, 3 }, .{ .alloc = allocator });
    defer out_a.deinit();
    var out_b = try Tensor.host(.f32, &.{ 3, 2 }, .{ .alloc = allocator });
    defer out_b.deinit();
    var out_c = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = allocator });
    defer out_c.deinit();

    const ev_a = try executor.download(outputs[0], out_a.host_data_mut());
    defer if (ev_a) |completion| executor.release_event(completion);
    const ev_b = try executor.download(outputs[1], out_b.host_data_mut());
    defer if (ev_b) |completion| executor.release_event(completion);
    const ev_c = try executor.download(outputs[2], out_c.host_data_mut());
    defer if (ev_c) |completion| executor.release_event(completion);

    if (ev_a) |completion| try executor.wait(completion);
    if (ev_b) |completion| try executor.wait(completion);
    if (ev_c) |completion| try executor.wait(completion);

    const got_a = out_a.as_const_slice(f32);
    const got_b = out_b.as_const_slice(f32);
    const got_c = out_c.as_const_slice(f32);
    std.debug.assert(got_a.len == 6);
    std.debug.assert(got_b.len == 6);
    std.debug.assert(got_c.len == 4);

    const expected_a = [_]f32{
        30.0, 38.0, 46.0,
        30.0, 38.0, 46.0,
    };
    const expected_b = [_]f32{
        10.0, 10.0,
        14.0, 14.0,
        18.0, 18.0,
    };
    const expected_c = [_]f32{
        62.0,  68.0,
        143.0, 158.0,
    };

    try expect_all_close("dA", got_a, &expected_a, 1e-4);
    try expect_all_close("dB", got_b, &expected_b, 1e-4);
    try expect_all_close("dC", got_c, &expected_c, 1e-4);

    log.info("OK: gradients match expected", .{});
}

pub fn run_train_demo(
    ctx: *zg.CompilationCtx,
    pipeline: *zg.Pipeline,
    warmup_steps: usize,
    steps: usize,
    quiet: bool,
) !void {
    const io = ctx.io;
    const allocator = ctx.allocator;

    const ParamsSpec = struct {
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        w3: Tensor,
        b3: Tensor,
    };

    const BatchSpec = struct {
        x: Tensor,
        y: Tensor,
    };

    const Fns = struct {
        fn loss(params: ParamsSpec, batch: BatchSpec) !Tensor {
            const bs_: i64 = 64;
            const h1_: i64 = 128;
            const h2_: i64 = 64;
            const out_: i64 = 10;

            const z1 = try batch.x.mm(params.w1);
            const b1b = try params.b1.broadcast_in_dim(&.{ bs_, h1_ }, &.{1});
            const a1 = try z1.add(b1b);

            const z2 = try a1.mm(params.w2);
            const b2b = try params.b2.broadcast_in_dim(&.{ bs_, h2_ }, &.{1});
            const a2 = try z2.add(b2b);

            const z3 = try a2.mm(params.w3);
            const b3b = try params.b3.broadcast_in_dim(&.{ bs_, out_ }, &.{1});
            const preds = try z3.add(b3b);
            const diff = try preds.sub(batch.y);
            const sq = try diff.mul(diff);
            return try sq.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
        }

        fn train_step(params: ParamsSpec, batch: BatchSpec) !struct { loss_val: Tensor, updated: ParamsSpec } {
            var vg = try zg.transforms.value_and_grad(loss, .{ params, batch }, .{});
            defer vg.deinit();
            var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
            defer params_tree.deinit();
            const optim = zg.optim.SGD{ .lr = 1e-2 };
            var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
            defer updated.deinit();
            return .{
                .loss_val = vg.outputs,
                .updated = try updated.extract(ParamsSpec),
            };
        }
    };

    const bs: i64 = 64;
    const in_dim: i64 = 784;
    const h1: i64 = 128;
    const h2: i64 = 64;
    const out_dim: i64 = 10;

    const params_spec = ParamsSpec{
        .w1 = Tensor.abstract(.f32, &.{ in_dim, h1 }),
        .b1 = Tensor.abstract(.f32, &.{h1}),
        .w2 = Tensor.abstract(.f32, &.{ h1, h2 }),
        .b2 = Tensor.abstract(.f32, &.{h2}),
        .w3 = Tensor.abstract(.f32, &.{ h2, out_dim }),
        .b3 = Tensor.abstract(.f32, &.{out_dim}),
    };
    const batch_spec = BatchSpec{
        .x = Tensor.abstract(.f32, &.{ bs, in_dim }),
        .y = Tensor.abstract(.f32, &.{ bs, out_dim }),
    };
    const inputs_spec = .{ params_spec, batch_spec };
    const donated = comptime zg.train.donated_input_indices(@TypeOf(inputs_spec), &.{0});

    const train = zg.train;
    var traced = try zg.trace(Fns.train_step, allocator, inputs_spec, .{ .name = "train_step" });
    defer traced.deinit();

    var exe = try pipeline.run(zg.Executor.LoadedProgram, &traced.program, ctx);
    defer exe.deinit();
    const executor = exe.executor;

    const true_w1 = try allocator.alloc(f32, @intCast(in_dim * h1));
    defer allocator.free(true_w1);
    const true_b1 = try allocator.alloc(f32, @intCast(h1));
    defer allocator.free(true_b1);
    const true_w2 = try allocator.alloc(f32, @intCast(h1 * h2));
    defer allocator.free(true_w2);
    const true_b2 = try allocator.alloc(f32, @intCast(h2));
    defer allocator.free(true_b2);
    const true_w3 = try allocator.alloc(f32, @intCast(h2 * out_dim));
    defer allocator.free(true_w3);
    const true_b3 = try allocator.alloc(f32, @intCast(out_dim));
    defer allocator.free(true_b3);

    fill_pattern(true_w1, 1e-6, 0.0);
    fill_pattern(true_b1, 1e-6, 0.0);
    fill_pattern(true_w2, 1e-6, 0.0);
    fill_pattern(true_b2, 1e-6, 0.0);
    fill_pattern(true_w3, 1e-6, 0.0);
    fill_pattern(true_b3, 1e-6, 0.0);

    var spec_tree = try zg.utils.Tree(Tensor).from(allocator, inputs_spec);
    defer spec_tree.deinit();

    var host_tensors = try spec_tree.map(Tensor, allocator, struct {
        fn f(alloc: std.mem.Allocator, spec: Tensor) !Tensor {
            return try Tensor.host(spec.dtype, spec.shape.const_slice(), .{ .alloc = alloc });
        }
    }.f);
    defer host_tensors.deinit_with(Tensor.deinit);

    for (host_tensors.leaves[0..6]) |*buf| fill_pattern(buf.as_slice(f32), 1e-7, 0.0);
    fill_inputs(host_tensors.leaves[6].as_slice(f32));

    const scratch1 = try allocator.alloc(f32, @intCast(bs * h1));
    defer allocator.free(scratch1);
    const scratch2 = try allocator.alloc(f32, @intCast(bs * h2));
    defer allocator.free(scratch2);
    try fill_targets(
        host_tensors.leaves[7],
        host_tensors.leaves[6],
        true_w1,
        true_b1,
        true_w2,
        true_b2,
        true_w3,
        true_b3,
        scratch1,
        scratch2,
        @intCast(bs),
        @intCast(in_dim),
        @intCast(h1),
        @intCast(h2),
        @intCast(out_dim),
    );

    var dev_tree = try host_tensors.map(Tensor, executor, struct {
        fn f(selected: *zg.Executor, tensor: Tensor) !Tensor {
            return try tensor.to_device(selected);
        }
    }.f);
    // TrainState releases the device tensors.
    defer dev_tree.deinit();

    const entry_function = traced.program.get_function_by_id(try traced.program.resolve_entry()) orelse return error.NoEntry;
    var state = try train.TrainState.init(
        allocator,
        exe,
        dev_tree.leaves,
        entry_function.returns.len,
        .{ .donated_input_indices = donated },
    );
    defer state.deinit(.all);

    for (0..warmup_steps) |_| {
        var result = try state.step();
        // TODO(pjrt): Verify that releasing an unawaited event is valid.
        if (result.event) |completion| executor.release_event(completion);
        if (!quiet) _ = try result.loss.item(f32);
        result.loss.deinit();
    }

    var loop_timer = zg.utils.LoopTimer{ .io = io, .label = "train-demo", .quiet = quiet };
    for (0..steps) |_| {
        try loop_timer.start_step();
        var result = try state.step();
        // TODO(pjrt): Verify that releasing an unawaited event is valid.
        if (result.event) |completion| executor.release_event(completion);
        loop_timer.mark("dispatch");

        const loss: ?f32 = if (quiet) null else try result.loss.item(f32);
        loop_timer.mark("sync+read");

        result.loss.deinit();
        loop_timer.mark("cleanup");

        loop_timer.end_step(loss);
    }

    log.info("avg_step_ms={d:.3}", .{loop_timer.avg_ms()});
    log.info("OK", .{});
}

/// Tunes, compiles, executes, and verifies the kernel-provider demo.
///
/// Providers discover PR candidates. Multi-provider epilogue runs use explicit
///  disjoint requests. Tuning populates the kernel store before runtime
///  preparation, PR kernelization, and PJRT execution.
pub const KernelProviderDemoOptions = struct {
    /// Matrix-product row count.
    m: i64,
    /// Matrix-product column count.
    n: i64,
    /// Matrix-product reduction size.
    k: i64,
    /// Element data type used by the candidate region.
    dtype: zg.pr.DType,
    /// Candidate graph family.
    pattern: KernelProviderDemoPattern,

    /// Label attached to output produced for the selected function.
    entry_label: []const u8,

    /// Optional destination and format for PR output.
    pr: ?zg.pr.dump.Config = null,

    /// Optional destination for the MLIR passed to PJRT.
    mlir: ?zg.output.Config = null,

    /// Optional destination for optimized HLO produced by PJRT.
    optimized_hlo: ?zg.output.Config = null,

    /// Optional destination for the kernelization report.
    kernels: ?zg.output.Target = null,
};

pub fn run_kernel_provider_demo(
    ctx: *zg.CompilationCtx,
    client: *zg.pjrt.Client,
    execution_template: *zg.pjrt.Execution,
    backend_template: *zg.pjrt.Backend,
    environ: *const std.process.Environ.Map,
    options: KernelProviderDemoOptions,
    provider_kinds: []const KernelProviderDemoKind,
) !void {
    const io = ctx.io;
    const allocator = ctx.allocator;
    const device = execution_template.device;

    const demo_cache = try zg.Cache.init(io, environ, .{});

    var tvm_dispatch: if (zg.build_options.has_tvm) zg.tvm.DispatchState else void = undefined;
    var tvm_impl: if (zg.build_options.has_tvm) zg.tvm.Provider else void = undefined;
    var has_tvm = false;

    if (kind_requested(provider_kinds, .tvm)) {
        if (comptime !zg.build_options.has_tvm) {
            log.err("tvm provider requested but binary was built without the TVM integration", .{});
            return error.TvmUnavailable;
        }
        try client.require_typed_ffi();
        const target_kind: zg.tvm.TargetKind = if (client.is_cuda()) .cuda else .cpu;
        const compile_config = try zg.tvm.CompileConfig.from_environ(
            environ,
            target_kind,
        );
        tvm_dispatch = zg.tvm.DispatchState.init(io, allocator, demo_cache);
        tvm_impl = try zg.tvm.Provider.init(
            io,
            demo_cache,
            &tvm_dispatch,
            .{
                .compile = compile_config,
                .max_trials = 8,
                .trials_per_iter = 4,
            },
        );
        has_tvm = true;
    }
    defer if (zg.build_options.has_tvm and has_tvm) tvm_dispatch.deinit();

    var mirage_dispatch: if (zg.build_options.has_mirage) zg.mirage.dispatch.MirageDispatchState else void = undefined;
    var mirage_impl: if (zg.build_options.has_mirage) zg.mirage.provider.MirageProvider else void = undefined;
    var has_mirage = false;

    if (kind_requested(provider_kinds, .mirage)) {
        if (comptime !zg.build_options.has_mirage) {
            log.err("mirage provider requested but binary was built without the Mirage integration", .{});
            return error.MirageUnavailable;
        }
        const mirage_config = zg.mirage.config.Config.from_environ(environ) catch |err| {
            log.err("Mirage configuration failed: {s}", .{@errorName(err)});
            return err;
        };
        mirage_dispatch = zg.mirage.dispatch.MirageDispatchState.init(
            allocator,
            mirage_config.compile,
        );
        mirage_impl = zg.mirage.provider.MirageProvider.init(
            &mirage_dispatch,
            .{
                .runtime = mirage_config.runtime,
                .search = .{
                    .preset = if (options.pattern == .mlp) .mlp else .default,
                    .time_limit_seconds = 30,
                },
            },
        ) catch |err| {
            mirage_dispatch.deinit();
            return err;
        };
        has_mirage = true;
    }
    defer if (zg.build_options.has_mirage and has_mirage) mirage_dispatch.deinit();

    var providers_buf: [2]zg.kernel.KernelProvider = undefined;
    var n_providers: usize = 0;
    for (provider_kinds) |kind| switch (kind) {
        .tvm => if (zg.build_options.has_tvm and has_tvm) {
            providers_buf[n_providers] = tvm_impl.kernel_provider();
            n_providers += 1;
        },
        .mirage => if (zg.build_options.has_mirage and has_mirage) {
            providers_buf[n_providers] = mirage_impl.kernel_provider();
            n_providers += 1;
        },
    };
    const providers = providers_buf[0..n_providers];

    var pnames_buf: [2][]const u8 = undefined;
    for (provider_kinds, 0..) |kind, i| pnames_buf[i] = @tagName(kind);
    const provider_names = pnames_buf[0..provider_kinds.len];

    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    const entry = try build_kernelized_demo_program(
        &program,
        provider_names,
        provider_names.len > 1,
        options.m,
        options.n,
        options.k,
        options.dtype,
        options.pattern,
    );
    try program.set_entry(entry);

    var candidates = zg.kernel.Candidates.init(allocator);
    defer candidates.deinit();
    var extracted = zg.kernel.ExtractedCandidates.init(allocator);
    defer extracted.deinit();

    var pipeline = zg.Pipeline.init(allocator);
    defer pipeline.deinit();
    try pipeline.add(zg.pr.Validate{});
    if (options.pr) |selected| {
        var config = selected;
        config.entry_label = config.entry_label orelse options.entry_label;
        try pipeline.add(zg.pr.dump.Dump{ .config = config });
    }
    try pipeline.add(zg.pr.transform.kernelize.DiscoverCandidates{
        .providers = providers,
        .candidates = &candidates,
    });
    try pipeline.add(zg.pr.transform.kernelize.ExtractCandidates{
        .candidates = &candidates,
        .extracted = &extracted,
    });
    _ = try pipeline.run(*zg.pr.Program, &program, ctx);

    var candidate_factory = zg.pjrt.CandidateExecutableFactory{
        .execution = execution_template,
        .providers = providers,
        .compile_options = backend_template.compiler.options,
    };
    var executable_measurer = zg.tune.ExecutableMeasurer{
        .io = io,
        .factory = candidate_factory.interface(),
    };
    var store = try zg.tune.tune(io, allocator, &program, providers, .{
        .device = execution_template.interface.device,
        .candidates = &extracted,
        .measurer = executable_measurer.interface(),
    });
    defer store.deinit();

    var dispatch_registry = zg.kernel.DispatchRegistry.init(allocator);
    defer dispatch_registry.deinit();
    try dispatch_registry.register_providers(providers);
    try dispatch_registry.prepare(&store, .{
        .device = execution_template.interface.device,
    });

    var report: ?zg.pr.transform.kernelize.Report = if (options.kernels != null)
        .init(allocator)
    else
        null;
    defer if (report) |*value| value.deinit();

    var kernelize = zg.pr.transform.kernelize.KernelizePass{
        .store = &store,
        .candidates = &extracted,
        .device = execution_template.interface.device,
        .report = if (report) |*value| value else null,
    };
    try pipeline.add(&kernelize);
    if (report) |*value| {
        try pipeline.add(zg.pr.transform.kernelize.DumpKernels{
            .report = value,
            .target = options.kernels.?,
        });
    }

    try pipeline.add(zg.pr.transform.outline.Pass{});
    try pipeline.add(zg.mlir.stablehlo.Lower{
        .config = .{
            .encoding = if (options.mlir == null) .binary else .text,
        },
    });
    if (options.mlir) |selected| {
        var config = selected;
        config.entry_label = config.entry_label orelse options.entry_label;
        try pipeline.add(zg.stablehlo.Dump{ .config = config });
    }

    var execution = try zg.pjrt.Execution.init(client, device, .{
        .store = &store,
        .dispatch_registry = &dispatch_registry,
    });
    var backend = zg.pjrt.Backend.init(&execution, backend_template.compiler.options);
    try pipeline.add(&backend.interface);
    if (options.optimized_hlo) |selected| {
        var config = selected;
        config.entry_label = config.entry_label orelse options.entry_label;
        try pipeline.add(zg.pjrt.DumpOptimizedHlo{
            .execution = &execution,
            .config = config,
        });
    }

    var exe = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        ctx,
    );
    defer exe.deinit();

    return try run_kernel_provider_demo_executable(
        allocator,
        io,
        exe,
        provider_kinds.len,
        options.m,
        options.n,
        options.k,
        options.dtype,
        options.pattern,
    );
}

fn kind_requested(kinds: []const KernelProviderDemoKind, target: KernelProviderDemoKind) bool {
    for (kinds) |k| if (k == target) return true;
    return false;
}

/// Execute the kernel provider demo program and verify results.
fn run_kernel_provider_demo_executable(
    allocator: std.mem.Allocator,
    io: std.Io,
    loaded_program: zg.Executor.LoadedProgram,
    n_providers: usize,
    m: i64,
    n: i64,
    k: i64,
    dtype: zg.pr.DType,
    pattern: KernelProviderDemoPattern,
) !void {
    const executor = loaded_program.executor;
    const a_shape = [_]i64{ m, k };
    const b_shape = [_]i64{ k, n };
    const c_shape = switch (pattern) {
        .epilogue => [_]i64{ m, n },
        .mlp => b_shape,
    };
    const output_shape = [_]i64{ m, n };

    var host_a = try Tensor.host(dtype, &a_shape, .{ .alloc = allocator });
    defer host_a.deinit();
    try fill_kernel_demo_tensor(
        &host_a,
        dtype,
        if (pattern == .mlp) 1.0 / @as(f32, @floatFromInt(k)) else 1.0,
    );
    var host_b = try Tensor.host(dtype, &b_shape, .{ .alloc = allocator });
    defer host_b.deinit();
    try fill_kernel_demo_tensor(&host_b, dtype, 1.0);
    var host_c = try Tensor.host(dtype, &c_shape, .{ .alloc = allocator });
    defer host_c.deinit();
    try fill_kernel_demo_tensor(&host_c, dtype, if (pattern == .mlp) 1.0 else 2.0);

    const dev_a = try executor.upload(host_a.host_data(), dtype, host_a.dims());
    defer executor.release(dev_a);
    const dev_b = try executor.upload(host_b.host_data(), dtype, host_b.dims());
    defer executor.release(dev_b);
    const dev_c = try executor.upload(host_c.host_data(), dtype, host_c.dims());
    defer executor.release(dev_c);

    var out_host = try Tensor.host(dtype, &output_shape, .{ .alloc = allocator });
    defer out_host.deinit();

    const t0 = std.Io.Timestamp.now(io, .awake);
    var outputs: [1]zg.Executor.Buffer = undefined;
    const event = try executor.invoke(
        loaded_program,
        &.{ dev_a, dev_b, dev_c },
        &outputs,
        .{},
    );
    defer if (event) |completion| executor.release_event(completion);
    defer executor.release(outputs[0]);

    if (try executor.download(outputs[0], out_host.host_data_mut())) |completion| {
        defer executor.release_event(completion);
        try executor.wait(completion);
    }
    const dur = t0.untilNow(io, .awake);
    log.info("Executed dur={f}", .{dur});

    const expected: f32 = switch (pattern) {
        .epilogue => 2.0 * @as(f32, @floatFromInt(n_providers)) *
            @as(f32, @floatFromInt(k)) + 4.0,
        .mlp => 1.0 / (1.0 + @exp(-1.0)),
    };
    try verify_kernel_demo_tensor(out_host, dtype, expected);
    log.info("OK: output matches expected ({d} provider(s))", .{n_providers});
}

fn fill_kernel_demo_tensor(tensor: *Tensor, dtype: zg.pr.DType, value: f32) !void {
    switch (dtype) {
        .f16 => @memset(tensor.as_slice(u16), zg.pr.DType.f16.encode(f32, value)),
        .bf16 => @memset(tensor.as_slice(u16), zg.pr.DType.bf16.encode(f32, value)),
        .f32 => @memset(tensor.as_slice(f32), value),
        else => return error.UnsupportedDType,
    }
}

fn verify_kernel_demo_tensor(tensor: Tensor, dtype: zg.pr.DType, expected: f32) !void {
    const tolerance: f32 = if (dtype == .f32) 1e-4 else 1e-2;
    const len = tensor.host_data().len / dtype.size_in_bytes();
    for (0..len) |index| {
        const value: f32 = switch (dtype) {
            .f16 => zg.pr.DType.f16.decode(f32, tensor.as_const_slice(u16)[index]),
            .bf16 => zg.pr.DType.bf16.decode(f32, tensor.as_const_slice(u16)[index]),
            .f32 => tensor.as_const_slice(f32)[index],
            else => return error.UnsupportedDType,
        };
        if (@abs(value - expected) > tolerance) {
            log.err("mismatch[{d}]: got {d}, expected {d}", .{ index, value, expected });
            return error.NumericalMismatch;
        }
    }
}

pub const KernelProviderDemoKind = enum {
    /// TVM TensorIR and MetaSchedule provider.
    tvm,
    /// Mirage symbolic graph-search provider.
    mirage,
};

pub const KernelProviderDemoPattern = enum {
    /// Matrix multiplication followed by elementwise arithmetic.
    epilogue,
    /// Two matrix multiplications joined by a SiLU gate.
    mlp,
};

pub fn print_pr(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    const forward = try build_demo_program(&program);

    const forward_func = program.get_function_by_id(forward) orelse unreachable;
    const derived = try zg.pr.ad.vjp(allocator, &program, forward, "main_vjp", .{});
    const vjp_func = program.get_function_by_id(derived) orelse unreachable;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const tc = truecolor_from_env(environ);
    try stdout.writeAll("=== Forward ===\n");
    try zg.pr.zxpr.emit(forward_func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .truecolor_auto = tc }));
    try stdout.writeAll("\n=== VJP ===\n");
    try zg.pr.zxpr.emit(vjp_func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .truecolor_auto = tc }));
}

/// Return true when `ZG_TRUECOLOR` is set to a non-empty value.
///
/// Library styling code receives the decision from a caller with an explicit
///  environment map.
fn truecolor_from_env(environ: *const std.process.Environ.Map) bool {
    const v = environ.get("ZG_TRUECOLOR") orelse return false;
    return v.len > 0;
}

/// Enumerate TVM FFI global functions.
/// Writes available operations to stdout.
pub fn dump_tvm_ffi_symbols(io: std.Io, allocator: std.mem.Allocator, load_compiler: bool) !void {
    if (comptime !zg.build_options.has_tvm) {
        log.err("TVM FFI symbol dump requires the opt-in TVM integration", .{});
        return error.TvmUnavailable;
    }
    const names = try zg.tvm.runtime.list_global_names(
        allocator,
        if (load_compiler) .compiler else .ffi,
    );
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }

    var stdout_buf: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    try out.print("# TVM FFI Global Functions (total: {d})\n#\n", .{names.len});

    var current_prefix: []const u8 = "";
    var category_count: usize = 0;
    for (names) |name| {
        const prefix = if (std.mem.indexOf(u8, name, ".")) |idx| name[0..idx] else "root";

        if (!std.mem.eql(u8, prefix, current_prefix)) {
            if (category_count > 0) try out.writeAll("\n");
            try out.print("## {s}\n", .{prefix});
            current_prefix = prefix;
            category_count += 1;
        }
        try out.print("  {s}\n", .{name});
    }
}

pub fn print_tvm_kernelize_pr(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    sweep_palettes: bool,
    palette: ?zg.pr.zxpr.style.Palette,
) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var builder = try zg.pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const a = try Tensor.param(&builder, .f32, &.{ 2, 3 });
    const b_t = try Tensor.param(&builder, .f32, &.{ 3, 2 });
    const c = try Tensor.param(&builder, .f32, &.{ 2, 2 });
    const d = try Tensor.param(&builder, .f32, &.{ 2, 2 });

    const pre = try c.add(d);

    try builder.push_region("tvm_matmul", &.{zg.kernel.provider_annotation("tvm")});
    const mm = try a.mm(b_t);
    try builder.pop_region();

    try builder.push_region("tvm_fused", &.{
        zg.kernel.provider_annotation("tvm"),
        zg.pr.transform.outline.annotation,
    });
    const sum = try mm.add(pre);
    const mul = try sum.mul(c);
    try builder.pop_region();

    const out = try mul.add(d);
    const out_var = try out.get_var();
    const func_result = try builder.finish(.{ .returns = &.{out_var} });
    const identity = try program.add_function(func_result);
    const func = program.get_function_by_id(identity) orelse unreachable;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const tc = truecolor_from_env(environ);
    if (sweep_palettes) {
        const palettes = [_]zg.pr.zxpr.style.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("=== Kernelize(TVM subgraph) [{s}] ===\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal, .truecolor_auto = tc }));
            try stdout.writeAll("\n");
        }
    } else {
        try stdout.writeAll("=== Kernelize(TVM subgraph) ===\n");
        try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = palette, .truecolor_auto = tc }));
    }
}

fn expect_all_close(label: []const u8, got: []const f32, expected: []const f32, tol: f32) !void {
    if (got.len != expected.len) return error.LengthMismatch;
    for (got, 0..) |v, i| {
        const diff = @abs(v - expected[i]);
        if (diff > tol) {
            std.log.err("{s} mismatch[{d}]: got {d}, expected {d}", .{ label, i, v, expected[i] });
            return error.NumericalMismatch;
        }
    }
}

fn fill_pattern(slice: []f32, scale: f32, offset: f32) void {
    for (slice, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 1024));
        v.* = offset + scale * base;
    }
}

/// Builds the selected kernel-provider workload.
///
/// `epilogue` returns one matrix product per provider, summed with `c`, then
///  multiplied by `c`. Multi-provider runs use explicit disjoint requests.
/// `mlp` returns two matrix products joined by a SiLU gate.
fn build_kernelized_demo_program(
    program: *zg.pr.Program,
    provider_names: []const []const u8,
    explicit_requests: bool,
    m: i64,
    n: i64,
    k: i64,
    dtype: zg.pr.DType,
    pattern: KernelProviderDemoPattern,
) !zg.pr.FunctionId {
    var b = try zg.pr.FunctionBuilder.init(program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(dtype, &.{ m, k });
    const b_id = try b.param_tensor(dtype, &.{ k, n });
    const c_id = try b.param_tensor(dtype, switch (pattern) {
        .epilogue => &.{ m, n },
        .mlp => &.{ k, n },
    });

    if (pattern == .mlp) {
        const first = try b.mm(a_id, b_id);
        const second = try b.mm(a_id, c_id);
        const activation = try b.logistic(first);
        const silu = try b.multiply(first, activation);
        const output = try b.multiply(silu, second);
        const func = try b.finish(.{ .returns = &.{output} });
        return try program.add_function(func);
    }

    // Region names share the program lifetime of their function references.
    const first_name = try (b.alloc()).print("{s}_region_0", .{provider_names[0]});
    if (explicit_requests) try b.push_region(first_name, &.{zg.kernel.provider_annotation(provider_names[0])});
    var acc_id = try b.mm(a_id, b_id);
    if (explicit_requests) try b.pop_region();

    for (provider_names[1..], 1..) |pname, i| {
        const rn = try (b.alloc()).print("{s}_region_{d}", .{ pname, i });
        if (explicit_requests) try b.push_region(rn, &.{zg.kernel.provider_annotation(pname)});
        const mm_id = try b.mm(a_id, b_id);
        if (explicit_requests) try b.pop_region();
        acc_id = try b.add(acc_id, mm_id);
    }

    const add_id = try b.add(acc_id, c_id);
    const out_id = try b.multiply(add_id, c_id);

    const func = try b.finish(.{ .returns = &.{out_id} });
    return try program.add_function(func);
}

fn fill_inputs(x: []f32) void {
    for (x, 0..) |*v, i| {
        const base = @as(f32, @floatFromInt(i % 256));
        v.* = base / 255.0;
    }
}

fn fill_targets(
    host_y: Tensor,
    host_x: Tensor,
    w1: []const f32,
    b1: []const f32,
    w2: []const f32,
    b2: []const f32,
    w3: []const f32,
    b3: []const f32,
    scratch1: []f32,
    scratch2: []f32,
    bs: usize,
    in_dim: usize,
    h1: usize,
    h2: usize,
    out_dim: usize,
) !void {
    if (host_y.dtype != .f32 or host_x.dtype != .f32) return error.UnsupportedDType;
    if (host_x.shape.len != 2 or host_y.shape.len != 2) return error.ShapeMismatch;
    const x_dims = host_x.shape.const_slice();
    const y_dims = host_y.shape.const_slice();
    if (x_dims[0] != @as(i64, @intCast(bs)) or x_dims[1] != @as(i64, @intCast(in_dim))) return error.ShapeMismatch;
    if (y_dims[0] != @as(i64, @intCast(bs)) or y_dims[1] != @as(i64, @intCast(out_dim))) return error.ShapeMismatch;

    if (w1.len != in_dim * h1 or b1.len != h1) return error.ShapeMismatch;
    if (w2.len != h1 * h2 or b2.len != h2) return error.ShapeMismatch;
    if (w3.len != h2 * out_dim or b3.len != out_dim) return error.ShapeMismatch;
    if (scratch1.len != bs * h1 or scratch2.len != bs * h2) return error.ShapeMismatch;

    const x = host_x.as_slice(f32);
    const y = host_y.as_slice(f32);

    var n: usize = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < h1) : (j += 1) {
            var acc: f32 = b1[j];
            var k: usize = 0;
            while (k < in_dim) : (k += 1) {
                acc += x[n * in_dim + k] * w1[k * h1 + j];
            }
            scratch1[n * h1 + j] = acc;
        }
    }

    n = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < h2) : (j += 1) {
            var acc: f32 = b2[j];
            var k: usize = 0;
            while (k < h1) : (k += 1) {
                acc += scratch1[n * h1 + k] * w2[k * h2 + j];
            }
            scratch2[n * h2 + j] = acc;
        }
    }

    n = 0;
    while (n < bs) : (n += 1) {
        var j: usize = 0;
        while (j < out_dim) : (j += 1) {
            var acc: f32 = b3[j];
            var k: usize = 0;
            while (k < h2) : (k += 1) {
                acc += scratch2[n * h2 + k] * w3[k * out_dim + j];
            }
            y[n * out_dim + j] = acc;
        }
    }
}

/// Prints PR for an annotated attention pattern.
///
/// The region covers scaled dot-product attention for TVM kernelization.
pub fn print_tvm_attention_pr(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    sweep_palettes: bool,
    palette: ?zg.pr.zxpr.style.Palette,
) !void {
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();

    var builder = try zg.pr.FunctionBuilder.init(&program, "attention");
    defer builder.deinit();

    const batch: i64 = 2;
    const seq: i64 = 4;
    const head_dim: i64 = 64;

    const q = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });
    const k = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });
    const v = try Tensor.param(&builder, .f32, &.{ batch, seq, head_dim });

    // The annotation presents the complete attention block to TVM.
    try builder.push_region("attention", &.{zg.kernel.provider_annotation("tvm")});

    // Contracting the head dimension produces [batch, query, key] scores.
    const scores = try q.dot_general(k, .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{2},
    });

    const scale_val = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const scaled = try scores.mul(try Tensor.constant_like(scores, scale_val));

    const rank = scaled.rank();
    const axis: i64 = @intCast(rank - 1);
    const max_val = try scaled.reduce(.{ .axes = &.{axis}, .operation = .maximum });

    // Subtracting each row maximum stabilizes the softmax exponentials.
    const max_broadcast = try max_val.broadcast_in_dim(scaled.dims(), &.{ 0, 1 });
    const shifted = try scaled.sub(max_broadcast);
    const exp_vals = try shifted.exp();
    const sum_exp = try exp_vals.reduce(.{ .axes = &.{axis}, .operation = .sum });
    const sum_broadcast = try sum_exp.broadcast_in_dim(exp_vals.dims(), &.{ 0, 1 });
    const attn_weights = try exp_vals.div(sum_broadcast);

    // Contracting the key dimension produces [batch, query, head dimension].
    const out = try attn_weights.bmm(v);

    try builder.pop_region();

    const out_var = try out.get_var();
    const func_result = try builder.finish(.{ .returns = &.{out_var} });
    const identity = try program.add_function(func_result);
    const func = program.get_function_by_id(identity) orelse unreachable;

    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("Flush failed");

    const tc = truecolor_from_env(environ);
    if (sweep_palettes) {
        const palettes = [_]zg.pr.zxpr.style.Palette{ .default, .alt, .nord, .gruvbox_material, .flat_dark, .catppuccin, .tokyonight };
        for (palettes) |pal| {
            try stdout.print("\n==== Palette: {s} ====\n", .{@tagName(pal)});
            try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal, .truecolor_auto = tc }));
        }
    } else {
        const pal = palette orelse .default;
        try zg.pr.zxpr.emit(func, stdout, zg.pr.zxpr.style.config(.auto_stdout, .{ .palette = pal, .truecolor_auto = tc }));
    }
}
