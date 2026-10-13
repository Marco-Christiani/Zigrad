//! LLaMA checkpoint loading, tracing, compilation, and execution.

const std = @import("std");
const zg = @import("zigrad");

const model = @import("model.zig");
const Tensor = zg.Tensor;
const Backend = zg.Backend(zg.stablehlo.Artifact);

const log = std.log.scoped(.@"zg/llama_training");

/// Create an iota tensor from a traced tensor's builder.
///
/// TODO(api): Expose this operation through `Tensor`.
fn iota_from(t: Tensor, out_dtype: zg.DType, out_dims: []const i64, iota_dim: i64) !Tensor {
    const b = t.backing.traced.builder;
    return Tensor.from_var(b, try b.iota(out_dtype, out_dims, iota_dim));
}

const num_layers: usize = 16;

/// Per-attention-head dimension.
///
/// This value matches the LLaMA 3.2 1B checkpoint used by the example.
///
/// TODO(example): Read model dimensions from the checkpoint configuration.
const head_dim: i64 = 64;

/// Checkpoint-shaped model parameters.
const LlamaParams = model.LlamaWeights(num_layers);

const Batch = struct {
    x: Tensor,
    target_ids: Tensor,
    attention_mask: Tensor,
    mask: Tensor,
    sin: Tensor,
    cos: Tensor,
};

/// Program compiled and executed by the example.
pub const Mode = enum {
    training,
    inference,
};

/// Inputs controlling checkpoint loading and execution.
pub const Options = struct {
    /// Select the compiled program.
    mode: Mode = .training,
    /// Element type used for model parameters and activations.
    dtype: zg.DType = .bf16,
    /// Number of tokens in each batch row.
    sequence_length: i64 = 4,
    /// Number of batch rows.
    batch: i64 = 1,
    /// Number of untimed executions before measurement.
    warmup_steps: usize = 5,
    /// Number of measured executions.
    steps: usize = 20,
    /// Suppress per-step timing output.
    quiet: bool = false,
    /// SafeTensors checkpoint whose keys match `LlamaParams`.
    weights_path: []const u8,
    /// Checked CPU observation request whose input outlives execution.
    observation: ?ObservationContext = null,
};

const TrainStepResult = struct { loss_val: Tensor, updated: LlamaParams };

fn train_step_fn(params: LlamaParams, batch: Batch) !TrainStepResult {
    var vg = try zg.transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
    defer vg.deinit();
    return try sgd_step(params, &vg, @bitCast(checked_workload.learning_rate_bits));
}

fn sgd_step(params: LlamaParams, vg: *zg.transforms.ValueAndGrad(Tensor), lr: f32) !TrainStepResult {
    var params_tree = try zg.utils.Tree(Tensor).from(vg.grads.allocator, params);
    defer params_tree.deinit();
    const optim = zg.optim.SGD{ .lr = lr };
    var updated = try params_tree.map2(Tensor, &vg.grads, Tensor, optim, zg.optim.SGD.update);
    defer updated.deinit();
    return .{
        .loss_val = vg.outputs,
        .updated = try updated.extract(LlamaParams),
    };
}

fn loss_fn(params: LlamaParams, batch: Batch) !Tensor {
    const batch_size: i64 = batch.x.dims()[0];
    const seq: i64 = batch.x.dims()[1];
    const logits = try params.model.forward(
        batch.x,
        batch.mask,
        batch.attention_mask,
        batch.sin,
        batch.cos,
        1e-6,
    );
    // accumulate loss in f32 bc bf16 logits over the full vocab can overflow in logsumexp
    const logits_f = try logits.convert(.f32);
    const attn_mask = try batch.attention_mask.convert(logits_f.dtype);

    // gather one vocab entry per (batch, seq) row without flattening
    const row_b = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 0);
    const row_s = try iota_from(logits_f, .i32, &.{ batch_size, seq }, 1);
    const tgt_ids = try batch.target_ids.convert(.i32);
    const row_b3 = try row_b.reshape(&.{ batch_size, seq, 1 });
    const row_s3 = try row_s.reshape(&.{ batch_size, seq, 1 });
    const tgt_ids3 = try tgt_ids.reshape(&.{ batch_size, seq, 1 });
    const gather_idx = try row_b3.concatenate(&.{ row_s3, tgt_ids3 }, 2);
    const gather_params: zg.pr.GatherParams = .{
        .slice_sizes = &.{ 1, 1, 1 },
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{ 0, 1, 2 },
        .start_index_map = &.{ 0, 1, 2 },
        .index_vector_dim = 2,
    };
    const target_logits_2d = try logits_f.gather(gather_idx, gather_params);

    const max_logits = try logits_f.reduce(.{ .axes = &.{2}, .operation = .maximum });
    const max_b = try max_logits.broadcast_in_dim(logits_f.dims(), &.{ 0, 1 });
    const shifted = try logits_f.sub(max_b);

    // jax-style logsumexp lowering: shift in f32, exp in the model dtype (bf16 round-trip
    //  preserves jax numerics), accumulate in f32
    const shifted_f32 = try shifted.convert(.f32);
    const exp_logits = try (try shifted_f32.convert(logits.dtype)).exp();
    const exp_logits_f32 = try exp_logits.convert(.f32);
    const sum_exp = try exp_logits_f32.reduce(.{ .axes = &.{2}, .operation = .sum });
    const log_sum = try sum_exp.log();
    const logsumexp = try log_sum.add(try max_logits.convert(log_sum.dtype));

    const loss_per = try logsumexp.sub(try target_logits_2d.convert(logsumexp.dtype));
    const loss_per_out = try loss_per.convert(logits_f.dtype);

    const zero_b = try Tensor.constant_like(attn_mask, 0.0);
    const attn_zero = try attn_mask.compare(zero_b, .{ .direction = .GT, .compare_type = .FLOAT });
    const masked = try loss_per_out.select(attn_zero, zero_b);
    const masked_f = try masked.convert(.f32);
    const loss_sum = try masked_f.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });

    // Normalize by token count: loss / max(mask.sum(), 1)
    const attn_zero_f = try attn_zero.convert(.f32);
    const mask_count = try attn_zero_f.reduce(.{ .axes = &.{ 0, 1 }, .operation = .sum });
    const one = try Tensor.constant_like(mask_count, 1.0);
    const denom = try mask_count.max(one);
    const loss_norm = try loss_sum.div(denom);

    return loss_norm;
}

/// Load a checkpoint, compile the selected program, and execute it.
pub fn run(
    /// Compilation services and selected device.
    ctx: *zg.CompilationCtx,
    /// Terminal backend used after the shared PR to StableHLO lowering.
    backend: *Backend,
    /// Checkpoint, shape, and iteration configuration.
    options: Options,
) !void {
    const io = ctx.io;
    const allocator = ctx.allocator;
    if (options.batch <= 0 or options.sequence_length <= 0)
        return error.InvalidShape;

    var pipeline = zg.Pipeline.init(allocator);
    defer pipeline.deinit();
    try zg.mlir.stablehlo.pipeline.add(&pipeline, .{});
    try pipeline.add(backend);

    const training_mode = options.mode == .training;
    const model_dtype = options.dtype;
    const batch_size: i64 = if (options.observation) |o| o.input.parsed.value.workload.batch_size else options.batch;
    const seq: i64 = if (options.observation) |o| o.input.parsed.value.workload.sequence_length else options.sequence_length;
    log.info("mode={t} dtype={t} batch={d} sequence_length={d}", .{
        options.mode,
        model_dtype,
        batch_size,
        seq,
    });

    const mmap_data = if (options.observation) |observation| observation.input.checkpoint_bytes else zg.utils.mmap_file(options.weights_path) catch |err| {
        log.err("failed to mmap checkpoint at '{s}': {s}", .{ options.weights_path, @errorName(err) });
        return err;
    };
    defer if (options.observation == null) zg.utils.munmap(mmap_data);

    var st_file = try zg.SafetensorsFile.deserialize(mmap_data, allocator);
    defer st_file.deinit();

    const host_params = try zg.from_safetensors(LlamaParams, &st_file, .{
        .allocator = allocator,
        .dtype = model_dtype,
    });
    if (!options.quiet) log.info("loaded weights from {s}", .{options.weights_path});

    // batch host tensors, allocated directly from the known shapes
    // sin/cos tables hold the rotary frequencies for `head_dim/2` channels per
    //  position, so their second dim is derived from `head_dim`
    const rope_half = @divExact(head_dim, 2);
    const dims_seq_seq: [2]i64 = .{ seq, seq };
    const dims_seq_half: [2]i64 = .{ seq, rope_half };
    const dims_b_s: [2]i64 = .{ batch_size, seq };
    const host_batch = Batch{
        .x = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .target_ids = try Tensor.host(.i32, &dims_b_s, .{ .alloc = allocator }),
        .attention_mask = try Tensor.host(model_dtype, &dims_b_s, .{ .alloc = allocator }),
        .mask = try Tensor.host(model_dtype, &dims_seq_seq, .{ .alloc = allocator }),
        .sin = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
        .cos = try Tensor.host(model_dtype, &dims_seq_half, .{ .alloc = allocator }),
    };

    const inputs = .{ host_params, host_batch };

    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    var preparation_timer = zg.utils.LoopTimer{ .io = io, .label = "llama-preparation", .quiet = true };
    try preparation_timer.start_step();
    const entry = if (training_mode)
        try zg.trace_into(train_step_fn, allocator, &program, inputs, "main")
    else
        try zg.trace_into(loss_fn, allocator, &program, inputs, "main");
    try program.set_entry(entry);
    preparation_timer.mark("trace");
    const trace_elapsed_ns = preparation_timer.phases[0].ns;
    try preparation_timer.start_step();
    var loaded_program = try pipeline.run(zg.Executor.LoadedProgram, &program, ctx);
    preparation_timer.mark("pipeline-preparation");
    const pipeline_preparation_elapsed_ns = preparation_timer.phases[0].ns;
    defer loaded_program.deinit();
    const executor = loaded_program.executor;

    const donated = comptime zg.train.donated_input_indices(@TypeOf(inputs), &.{0});

    if (options.observation) |observation| {
        try fill_checked_batch(host_batch, observation.input.*);
    } else {
        // Fill batch leaves with synthetic inputs.
        const token_seed = [_]usize{ 128000, 128009, 128001, 128008 };
        const target_seed = [_]usize{ 128009, 128001, 128008, 128001 };
        var tokens: [token_seed.len]usize = undefined;
        var targets: [target_seed.len]usize = undefined;
        // Vocab from the loaded embedding shape (torch layout `[vocab, hidden]`).
        const vocab_usize: usize = @intCast(host_params.model.embed_tokens.weight.dims()[0]);
        for (token_seed, 0..) |value, idx| tokens[idx] = value % vocab_usize;
        for (target_seed, 0..) |value, idx| targets[idx] = value % vocab_usize;
        const batch_usize: usize = @intCast(batch_size);
        const seq_usize: usize = @intCast(seq);

        fill_i32_tokens(host_batch.x.as_slice(i32), batch_usize, seq_usize, &tokens);
        fill_i32_tokens(host_batch.target_ids.as_slice(i32), batch_usize, seq_usize, &targets);
        fill_attention_mask(host_batch.attention_mask, batch_usize, seq_usize, tokens.len);
        fill_causal_mask(host_batch.mask, seq_usize);
        fill_rope_tables(host_batch.sin, host_batch.cos, seq_usize, @intCast(head_dim));
    }

    // Combined (params, batch) host tree. Built by flattening the two concrete
    //  structs into leaf copies. `host_tree` becomes the sole owner of every
    //  host tensor: its `deinit_with(Tensor.deinit)` frees all host-backing
    //  allocations, and we must not also deinit `host_params`/`host_batch`
    //  directly (value copies share backing storage).
    var host_tree = try zg.utils.Tree(Tensor).from(allocator, .{ host_params, host_batch });
    defer host_tree.deinit_with(Tensor.deinit);

    // transfer inputs to the selected device
    var dev_tree = try host_tree.map(Tensor, executor, struct {
        fn f(selected: *zg.Executor, tensor: Tensor) (zg.Executor.Error || error{UnsupportedAval})!Tensor {
            return try tensor.to_device(selected);
        }
    }.f);

    var loop_timer = zg.utils.LoopTimer{
        .io = io,
        .label = "llama-training",
        .quiet = options.quiet,
    };

    // Each branch releases `dev_tree` according to whether `TrainState` owns
    //  its tensor leaves.
    if (training_mode) {
        // TrainState.deinit releases the leaf buffers after donation swaps,
        //  this scope releases only the tree arrays and paths
        defer dev_tree.deinit();

        const function = program.get_function_by_id(entry) orelse return error.NoEntry;
        var state = try zg.train.TrainState.init(
            allocator,
            loaded_program,
            dev_tree.leaves,
            function.returns.len,
            .{ .donated_input_indices = donated, .loss_dtype = .f32 },
        );
        defer state.deinit(.all);

        if (options.observation) |observation| {
            if (!executor.device.platform.eql(.cpu)) return reject(.UnsupportedDevice, "loaded executor is not CPU");
            try observation.input_readiness.wait(observation.input_readiness.context, state.input_buffers);
            var parameter_tree = try zg.utils.Tree(Tensor).from(allocator, host_params);
            defer parameter_tree.deinit();
            const mapping = try build_parameter_mapping(allocator, &parameter_tree, &st_file);
            defer allocator.free(mapping);
            var capture = SampleCapture{ .allocator = allocator, .state = &state, .mapping = mapping };
            defer capture.deinit();
            const names_hash = try ordered_names_digest(allocator, mapping);
            defer allocator.free(names_hash);
            var observation_timer = zg.utils.LoopTimer{ .io = io, .label = "llama-observation", .quiet = true };
            const updates = try observe_updates(&state, &observation_timer, &capture);
            try write_observation(io, allocator, observation.output, .{ .result = .{ .success = .{
                .device = observation.device,
                .identity = observation.input.identity(),
                .losses = updates.losses,
                .parameter_samples = .{ .ordered_names_sha256 = names_hash, .initial = capture.initial.?, .updated = capture.updated.? },
                .timing = .{ .preparation = .{ .trace_elapsed_ns = trace_elapsed_ns, .pipeline_preparation_elapsed_ns = pipeline_preparation_elapsed_ns }, .first_call_elapsed_ns = updates.untimed[0].step_through_scalar_read_ns, .untimed_updates = updates.untimed, .timed_update = updates.timed },
            } } });
            return;
        }

        for (0..options.warmup_steps) |_| {
            var result = try state.step();
            errdefer result.loss.deinit();
            defer if (result.event) |completion| executor.release_event(completion);
            _ = try result.loss.item(f32);
            result.loss.deinit();
        }

        for (0..options.steps) |_| {
            try loop_timer.start_step();
            var result = try state.step();
            errdefer result.loss.deinit();
            defer if (result.event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            const loss = try result.loss.item(f32);
            loop_timer.mark("sync+read");

            result.loss.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(if (options.quiet) null else loss);
        }
    } else {
        defer dev_tree.deinit_with(Tensor.deinit);

        const input_buffers = try allocator.alloc(zg.Executor.Buffer, dev_tree.leaves.len);
        defer allocator.free(input_buffers);
        for (input_buffers, dev_tree.leaves) |*slot, t| {
            slot.* = try t.buffer();
        }

        var output_buffers: [1]zg.Executor.Buffer = undefined;

        for (0..options.warmup_steps) |_| {
            const event = try executor.invoke(
                loaded_program,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            // read the result to exercise host transfer before the timed loop
            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], .f32, &.{});
            errdefer loss_tensor.deinit();
            // `item` synchronizes the transfer.
            _ = try loss_tensor.item(f32);
            loss_tensor.deinit();
        }

        for (0..options.steps) |_| {
            try loop_timer.start_step();
            const event = try executor.invoke(
                loaded_program,
                input_buffers,
                &output_buffers,
                .{},
            );
            defer if (event) |completion| executor.release_event(completion);
            loop_timer.mark("dispatch");

            var loss_tensor = Tensor.from_buffer(executor, output_buffers[0], .f32, &.{});
            errdefer loss_tensor.deinit();
            const loss = try loss_tensor.item(f32);
            loop_timer.mark("sync+read");

            loss_tensor.deinit();
            loop_timer.mark("cleanup");

            loop_timer.end_step(if (options.quiet) null else loss);
        }
    }

    log.info("avg_step_ms={d:.3} (warmup={d} steps={d} sequence_length={d})", .{
        loop_timer.avg_ms(),
        options.warmup_steps,
        options.steps,
        seq,
    });
}

/// Fill each batch row with token IDs and zero padding.
///
/// Each row receives `tokens[0..min(seq, tokens.len)]`.
fn fill_i32_tokens(out: []i32, batch: usize, seq: usize, tokens: []const usize) void {
    @memset(out, 0);
    if (batch == 0 or seq == 0) return;
    for (0..batch) |b| {
        const row = out[b * seq ..][0..seq];
        for (tokens[0..@min(seq, tokens.len)], 0..) |t, i| row[i] = @intCast(t);
    }
}

/// Fill a `[batch, seq]` attention mask with `active_len` active positions.
fn fill_attention_mask(buf: Tensor, batch: usize, seq: usize, active_len: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode(f32, 0.0));
            const count = @min(seq, active_len);
            const one = tag.encode(f32, 1.0);
            for (0..batch) |bx| for (0..count) |i| {
                slice[bx * seq + i] = one;
            };
        },
        else => @panic("fill_attention_mask: unsupported dtype"),
    }
}

/// Fill a `[seq, seq]` lower-triangular causal mask.
fn fill_causal_mask(buf: Tensor, seq: usize) void {
    switch (buf.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const slice = buf.as_slice(T);
            @memset(slice, tag.encode(f32, 0.0));
            const one = tag.encode(f32, 1.0);
            for (0..seq) |i| for (0..i + 1) |j| {
                slice[i * seq + j] = one;
            };
        },
        else => @panic("fill_causal_mask: unsupported dtype"),
    }
}

/// Apply the LLaMA 3 three-region frequency correction in place.
fn llama3_rope_freq_correction(inv_freq: []f32) void {
    const factor: f32 = 32.0;
    const low_freq_factor: f32 = 1.0;
    const high_freq_factor: f32 = 4.0;
    const old_context_len: f32 = 8192.0;

    const low_freq_wavelen = old_context_len / low_freq_factor;
    const high_freq_wavelen = old_context_len / high_freq_factor;

    for (inv_freq) |*freq| {
        const wavelen = 2.0 * std.math.pi / freq.*;
        if (wavelen > low_freq_wavelen) {
            // Low-freq region: divide by factor.
            freq.* /= factor;
        } else if (wavelen >= high_freq_wavelen) {
            // Medium-freq region: smooth interpolation.
            const smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            const scaled = freq.* / factor;
            freq.* = (1.0 - smooth) * scaled + smooth * freq.*;
        }
        // High-freq region (wavelen < high_freq_wavelen): unchanged.
    }
}

/// Precompute RoPE sine and cosine tables with shape `[seq, head_dim / 2]`.
///
/// Inverse frequencies use f32 and a base frequency of 500000.0. The stored
///  values use the destination buffers' dtype.
fn fill_rope_tables(sin: Tensor, cos: Tensor, seq: usize, head_dim_runtime: usize) void {
    const half = head_dim_runtime / 2;

    // supported checkpoints have head dimensions up to 256
    var inv_freq: [128]f32 = undefined;
    std.debug.assert(half <= inv_freq.len);

    const base: f32 = 500000.0;
    for (0..half) |j| {
        const exp = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(head_dim_runtime));
        inv_freq[j] = 1.0 / std.math.pow(f32, base, exp);
    }
    llama3_rope_freq_correction(inv_freq[0..half]);

    switch (sin.dtype) {
        inline .f32, .bf16, .f16, .f64 => |tag| {
            const T = tag.StorageType();
            const s = sin.as_slice(T);
            const c = cos.as_slice(T);
            for (0..seq) |i| for (0..half) |j| {
                const theta = @as(f32, @floatFromInt(i)) * inv_freq[j];
                s[i * half + j] = tag.encode(f32, @sin(theta));
                c[i * half + j] = tag.encode(f32, @cos(theta));
            };
        },
        else => @panic("fill_rope_tables: unsupported dtype"),
    }
}

/// Fixed input and state policy for the checked CPU comparison.
pub const checked_workload = struct {
    pub const batch_revision = "zg/llama-3.2-1b-batch1-seq4/v1";
    pub const batch_size: u32 = 1;
    pub const sequence_length: u32 = 4;
    pub const learning_rate_bits: u32 = 0x38d1b717;
    pub const learning_rate_f32_bits = "0x38d1b717";
    pub const sample_extent = 16;
    pub const untimed_updates = 2;
    pub const framework_additional_warmup_steps: u32 = 1;
    pub const parameter_count = zg.utils.Tree(Tensor).leaf_count(LlamaParams);
    pub const parameter_names = blk: {
        @setEvalBranchQuota(20000);
        var names = zg.utils.meta.tree_paths(Tensor, LlamaParams);
        std.mem.sort([]const u8, &names, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.less);
        break :blk names;
    };
    pub const contract_sha256 = "16edcab5cfd2a1c7dcf4d40722b7d9b2cbce771c96dc160a19e59047e7f4a884";
    pub const specification_revision = "1a10e7335108c55bb858c8fb13d08b9a9414d1d0";
    pub const descriptor: ContentIdentity = .{ .byte_length = 2750, .sha256 = "eff0806d3f8024daa2790bc507e8668406dcfed38b2b3aee54419e3156770289" };
    pub const checkpoint: ContentIdentity = .{ .byte_length = 2471645608, .sha256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f" };
    pub const config: ContentIdentity = .{ .byte_length = 877, .sha256 = "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb" };
};

const TensorName = enum { tokens, targets, attention_mask, causal_mask, sin, cos };
const RawDType = enum { i32, bf16 };
const ConsumedDType = enum { i32, f32 };
const Encoding = enum { raw_le_row_major };
const Precision = enum { f32 };
const InputDevice = struct { kind: enum { cpu }, backend: enum { pjrt } };
const Specification = struct { id: []const u8, revision: []const u8, requirements: [2][]const u8 };
const ContentIdentity = struct { byte_length: u64, sha256: []const u8 };
const PayloadFile = struct {
    path: []const u8,
    byte_length: u64,
    sha256: []const u8,

    fn identity(self: PayloadFile) ContentIdentity {
        return .{ .byte_length = self.byte_length, .sha256 = self.sha256 };
    }
};
const InputTensor = struct {
    name: TensorName,
    dtype: RawDType,
    shape: []const i64,
    encoding: Encoding,
    path: []const u8,
    byte_length: u64,
    sha256: []const u8,
};
const SourceFile = struct { path: []const u8, sha256: []const u8 };
const Expected = struct {
    resident_revision: []const u8,
    benchmark_revision: []const u8,
    package_attribute: []const u8,
    configuration_key: []const u8,
    source_files: [6]SourceFile,
};
const InputManifest = struct {
    schema: []const u8,
    schema_version: u32,
    caller_contract_sha256: []const u8,
    performance_specification: Specification,
    precision: Precision,
    device: InputDevice,
    workload: struct { batch_revision: []const u8, batch_size: u32, sequence_length: u32 },
    descriptor: PayloadFile,
    model: struct { checkpoint: PayloadFile, config: PayloadFile, tied_embedding_output_head: bool },
    tensors: [6]InputTensor,
    training: struct { learning_rate_f32_bits: []const u8, sample_extent: u32, untimed_updates_before_timed: u32, timed_update_ordinal: u32 },
    expected: Expected,
};

const DescriptorTensor = struct { name: TensorName, dtype: RawDType, shape: []const i64, byte_length: u64, sha256: []const u8 };
const Descriptor = struct {
    format_version: u32,
    program: struct { source_repository: []const u8, source_revision: []const u8, artifact: ContentIdentity },
    checkpoint: struct { media_type: []const u8, byte_length: u64, sha256: []const u8 },
    model_config: struct { media_type: []const u8, byte_length: u64, sha256: []const u8 },
    semantics: struct {
        rms_norm_epsilon: f32,
        learning_rate: f32,
        attention_softmax_accumulator: []const u8,
        loss_accumulator: []const u8,
        loss_exponential_dtype: []const u8,
        loss_reduction: []const u8,
    },
    batch: struct {
        revision: []const u8,
        batch_size: u32,
        sequence_length: u32,
        encoding: []const u8,
        generation: struct {
            tokens: [4]i32,
            targets: [4]i32,
            active_tokens: u32,
            causal_mask: []const u8,
            rope: struct { head_dimension: u32, theta: u32, scaling_factor: u32, original_context_length: u32, low_frequency_factor: u32, high_frequency_factor: u32 },
        },
        tensors: [6]DescriptorTensor,
    },
};

pub const RefusalCode = enum {
    InvalidArgument,
    InvalidManifest,
    ManifestHashMismatch,
    MissingInput,
    InvalidPath,
    DescriptorMismatch,
    TensorHashMismatch,
    InvalidTensor,
    CheckpointMismatch,
    ConfigMismatch,
    UnsupportedBackend,
    UnsupportedDevice,
    UnsupportedSampleReadback,
    InvalidObservation,
    ExecutionFailed,
    OutputFailed,
};
pub const Refused = struct { code: RefusalCode, message: []const u8 };
pub const ObservedDevice = struct {
    kind: enum { cpu } = .cpu,
    backend: enum { pjrt } = .pjrt,
    platform_name: []const u8,
    device_kind: []const u8,
    ordinal: i32,
};
const TensorIdentity = struct { dtype: RawDType, byte_length: u64, sha256: []const u8 };
const ConsumedIdentity = struct { dtype: ConsumedDType, byte_length: u64, sha256: []const u8 };
const ObservedTensor = struct { name: TensorName, shape: []const i64, encoding: Encoding, raw: TensorIdentity, consumed: ConsumedIdentity };
const Identity = struct {
    input_manifest_sha256: []const u8,
    caller_contract_sha256: []const u8,
    performance_specification: Specification,
    batch_revision: []const u8,
    batch_size: u32,
    sequence_length: u32,
    descriptor: ContentIdentity,
    checkpoint: ContentIdentity,
    model_config: ContentIdentity,
    tied_embedding_output_head: bool,
    tensors: [6]ObservedTensor,
    expected: Expected,
};
const TrainingObservation = struct {
    learning_rate_f32_bits: []const u8 = checked_workload.learning_rate_f32_bits,
    sample_extent: u32 = checked_workload.sample_extent,
    framework_additional_warmup_steps: u32 = checked_workload.framework_additional_warmup_steps,
};
const Loss = struct { value_f32: f32, producer_ordinal: u32, evaluated_parameter_update_count: u32 };
const Losses = struct { initial: Loss, after_step: Loss };
const Sample = struct {
    name: []const u8,
    shape: []const i64,
    update_count: u32,
    values_f32: [checked_workload.sample_extent]f32,
    little_endian_f32_sha256: []const u8,
};
const ParameterSamples = struct {
    order: []const u8 = "checkpoint_name_utf8_bytewise",
    layout: []const u8 = "row_major_first16",
    ordered_names_encoding: []const u8 = "utf8_compact_json_string_array",
    ordered_names_sha256: []const u8,
    initial: []Sample,
    updated: []Sample,
};
const Reason = struct { reason: []const u8 };
const StageTiming = union(enum) { available: struct { elapsed_ns: u64 }, unavailable: Reason };
const ForwardTiming = union(enum) {
    executed: struct { elapsed_ns: u64, evaluated_parameter_update_count: u32, prepares_compilation_cache_state: bool },
    not_executed: Reason,
};
const Preparation = struct {
    trace_elapsed_ns: u64,
    pipeline_preparation_elapsed_ns: u64,
    pipeline_contents: [3][]const u8 = .{ "lowering", "compilation", "loading" },
    lowering: StageTiming = .{ .unavailable = .{ .reason = "combined_pipeline_boundary" } },
    compile: StageTiming = .{ .unavailable = .{ .reason = "combined_pipeline_boundary" } },
    load: StageTiming = .{ .unavailable = .{ .reason = "combined_pipeline_boundary" } },
};
const UpdateTiming = struct {
    ordinal: u32,
    input_update_count: u32,
    output_update_count: u32,
    loss_evaluated_update_count: u32,
    dispatch_ns: u64,
    scalar_read_sync_ns: u64,
    cleanup_ns: u64,
    step_through_scalar_read_ns: u64,
    full_step_ns: u64,
    cleanup_scope: []const u8 = "loss_tensor_deinit_and_completion_event_release",
};
const Timing = struct {
    unit: []const u8 = "ns",
    clock: []const u8 = "std.Io.Clock.awake",
    synchronization: []const u8 = "completion_event_wait_then_Tensor.item(f32)",
    preparation: Preparation,
    first_call_elapsed_ns: u64,
    first_call_boundary: []const u8 = "sgd1_dispatch_through_scalar_read_excluding_preparation_transfer_samples",
    initial_forward: ForwardTiming = .{ .not_executed = .{ .reason = "initial_loss_from_sgd1_pre_update_computation" } },
    post_update_forward: ForwardTiming = .{ .not_executed = .{ .reason = "state1_loss_from_sgd2_pre_update_computation" } },
    untimed_updates: [checked_workload.untimed_updates]UpdateTiming,
    timed_update: UpdateTiming,
};
pub const Success = struct {
    precision: Precision = .f32,
    device: ObservedDevice,
    identity: Identity,
    training: TrainingObservation = .{},
    losses: Losses,
    parameter_samples: ParameterSamples,
    timing: Timing,
};
pub const Observation = struct {
    schema: []const u8 = "zigrad.resident-llama-observation",
    schema_version: u32 = 1,
    result: union(enum) { success: Success, refused: Refused },
};

/// Checked transport paths. All three flags form one observation request.
pub const ObservationArgs = struct {
    input_manifest: []const u8,
    input_manifest_sha256: []const u8,
    output: []const u8,
};

fn reject(code: RefusalCode, message: []const u8) anyerror {
    if (@import("builtin").is_test) log.debug("{t}: {s}", .{ code, message }) else log.err("{t}: {s}", .{ code, message });
    return switch (code) {
        inline else => |tag| @field(anyerror, @tagName(tag)),
    };
}

pub fn refusal_code(err: anyerror) RefusalCode {
    return std.meta.stringToEnum(RefusalCode, @errorName(err)) orelse .ExecutionFailed;
}

fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn valid_hex(value: []const u8, length: usize) bool {
    if (value.len != length) return false;
    for (value) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

fn digest(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    const hex = std.fmt.bytesToHex(hash, .lower);
    return allocator.dupe(u8, &hex);
}

fn same_identity(a: ContentIdentity, b: ContentIdentity) bool {
    return a.byte_length == b.byte_length and equal(a.sha256, b.sha256);
}

fn validate_path(path: []const u8) !void {
    if (path.len == 0 or std.fs.path.isAbsolute(path) or !std.unicode.utf8ValidateSlice(path) or std.mem.indexOfScalar(u8, path, 0) != null)
        return reject(.InvalidPath, "payload path must be relative UTF-8");
    var components = std.mem.splitScalar(u8, path, '/');
    while (components.next()) |component| {
        if (component.len == 0 or equal(component, ".") or equal(component, ".."))
            return reject(.InvalidPath, "payload path has an empty or escaping component");
    }
}

fn shape_elements(shape: []const i64) !u64 {
    var count: u64 = 1;
    for (shape) |dim| {
        if (dim <= 0) return reject(.InvalidTensor, "tensor dimensions must be positive");
        count = std.math.mul(u64, count, @intCast(dim)) catch return reject(.InvalidTensor, "shape product overflows");
    }
    return count;
}

fn tensor_shape(name: TensorName, batch: u32, seq: u32) [2]i64 {
    return switch (name) {
        .tokens, .targets, .attention_mask => .{ batch, seq },
        .causal_mask => .{ seq, seq },
        .sin, .cos => .{ seq, @divExact(head_dim, 2) },
    };
}

fn validate_input_manifest(m: InputManifest) !void {
    if (!equal(m.schema, "zigrad.resident-llama-input") or m.schema_version != 1)
        return reject(.InvalidManifest, "unsupported input schema/version");
    const w = m.workload;
    if (!equal(w.batch_revision, checked_workload.batch_revision) or w.batch_size != checked_workload.batch_size or w.sequence_length != checked_workload.sequence_length)
        return reject(.InvalidManifest, "workload differs from the checked comparison");
    if (!equal(m.caller_contract_sha256, checked_workload.contract_sha256) or
        !equal(m.performance_specification.id, "spruce-meridian-cedar") or
        !equal(m.performance_specification.revision, checked_workload.specification_revision) or
        !equal(m.performance_specification.requirements[0], "OD-010") or !equal(m.performance_specification.requirements[1], "QR-003"))
        return reject(.InvalidManifest, "caller/specification identity mismatch");
    if (!same_identity(m.descriptor.identity(), checked_workload.descriptor)) return reject(.DescriptorMismatch, "descriptor identity mismatch");
    if (!same_identity(m.model.checkpoint.identity(), checked_workload.checkpoint)) return reject(.CheckpointMismatch, "checkpoint identity mismatch");
    if (!same_identity(m.model.config.identity(), checked_workload.config) or !m.model.tied_embedding_output_head)
        return reject(.ConfigMismatch, "model configuration or tied head mismatch");
    try validate_path(m.descriptor.path);
    try validate_path(m.model.checkpoint.path);
    try validate_path(m.model.config.path);
    if (!equal(m.training.learning_rate_f32_bits, checked_workload.learning_rate_f32_bits) or
        m.training.sample_extent != checked_workload.sample_extent or m.training.untimed_updates_before_timed != checked_workload.untimed_updates or
        m.training.timed_update_ordinal != checked_workload.untimed_updates + 1)
        return reject(.InvalidManifest, "training policy mismatch");
    var seen: [6]bool = @splat(false);
    for (m.tensors) |t| {
        const index = @backingInt(t.name);
        if (seen[index]) return reject(.InvalidTensor, "duplicate tensor name");
        seen[index] = true;
        try validate_path(t.path);
        if (!valid_hex(t.sha256, 64)) return reject(.InvalidManifest, "tensor SHA256 must be lowercase hexadecimal");
        const shape = tensor_shape(t.name, w.batch_size, w.sequence_length);
        const dtype: RawDType = switch (t.name) {
            .tokens, .targets => .i32,
            else => .bf16,
        };
        const elements = try shape_elements(t.shape);
        const length = std.math.mul(u64, elements, @as(u64, if (dtype == .i32) 4 else 2)) catch return reject(.InvalidTensor, "tensor byte length overflows");
        if (t.dtype != dtype or !std.mem.eql(i64, t.shape, &shape) or length != t.byte_length)
            return reject(.InvalidTensor, "tensor dtype/shape/length mismatch");
    }
    const expected_paths = .{
        "examples/llama-training/src/main.zig", "examples/llama-training/src/training.zig", "src/utils/loop_timer.zig",
        "nix/flake/examples.nix",               "nix/flake/configuration-definitions.nix",  "examples/llama-training/build.zig",
    };
    if (!valid_hex(m.expected.resident_revision, 40) or !valid_hex(m.expected.benchmark_revision, 40) or
        !equal(m.expected.package_attribute, "zigrad-example-llama-training-cpu") or !equal(m.expected.configuration_key, "xla:cpu"))
        return reject(.InvalidManifest, "expected revision/package/configuration is invalid");
    inline for (expected_paths, 0..) |path, i| {
        if (!equal(m.expected.source_files[i].path, path) or !valid_hex(m.expected.source_files[i].sha256, 64))
            return reject(.InvalidManifest, "expected source set/hash is invalid");
    }
}

fn parse_manifest(allocator: std.mem.Allocator, bytes: []const u8, expected_hash: []const u8) !std.json.Parsed(InputManifest) {
    if (!valid_hex(expected_hash, 64)) return reject(.InvalidArgument, "manifest SHA256 must be lowercase hexadecimal");
    const hash = try digest(allocator, bytes);
    defer allocator.free(hash);
    if (!equal(hash, expected_hash)) return reject(.ManifestHashMismatch, "manifest bytes differ from supplied SHA256");
    const parsed = std.json.parseFromSlice(InputManifest, allocator, bytes, .{ .allocate = .alloc_always, .duplicate_field_behavior = .@"error", .ignore_unknown_fields = false }) catch
        return reject(.InvalidManifest, "manifest JSON is malformed or incomplete");
    errdefer parsed.deinit();
    try validate_input_manifest(parsed.value);
    return parsed;
}

/// Manifest storage and observed consumed identities live until `deinit`.
pub const CheckedInput = struct {
    parsed: std.json.Parsed(InputManifest),
    manifest_sha256: []const u8,
    checkpoint_path: [:0]const u8,
    checkpoint_bytes: []align(std.heap.page_size_min) u8,
    tensors: [6][]const u8,
    observed: [6]ObservedTensor,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CheckedInput) void {
        for (self.tensors) |bytes| self.allocator.free(bytes);
        for (self.observed) |t| self.allocator.free(t.consumed.sha256);
        zg.utils.munmap(self.checkpoint_bytes);
        self.allocator.free(self.checkpoint_path);
        self.allocator.free(self.manifest_sha256);
        self.parsed.deinit();
    }

    fn identity(self: CheckedInput) Identity {
        const m = self.parsed.value;
        return .{
            .input_manifest_sha256 = self.manifest_sha256,
            .caller_contract_sha256 = m.caller_contract_sha256,
            .performance_specification = m.performance_specification,
            .batch_revision = m.workload.batch_revision,
            .batch_size = m.workload.batch_size,
            .sequence_length = m.workload.sequence_length,
            .descriptor = m.descriptor.identity(),
            .checkpoint = m.model.checkpoint.identity(),
            .model_config = m.model.config.identity(),
            .tied_embedding_output_head = m.model.tied_embedding_output_head,
            .tensors = self.observed,
            .expected = m.expected,
        };
    }
};

fn payload_path(io: std.Io, allocator: std.mem.Allocator, root: []const u8, path: []const u8) ![:0]const u8 {
    try validate_path(path);
    const joined = try std.fs.path.join(allocator, &.{ root, path });
    defer allocator.free(joined);
    const resolved = std.Io.Dir.cwd().realPathFileAlloc(io, joined, allocator) catch return reject(.MissingInput, "payload file is missing or inaccessible");
    errdefer allocator.free(resolved);
    if (!std.mem.startsWith(u8, resolved, root) or resolved.len <= root.len or (root.len > 1 and resolved[root.len] != '/'))
        return reject(.InvalidPath, "payload symlink escapes manifest directory");
    return resolved;
}

fn verify_bytes(allocator: std.mem.Allocator, bytes: []const u8, identity: ContentIdentity, code: RefusalCode) !void {
    if (bytes.len != identity.byte_length) return reject(code, "file length differs from declared bytes");
    const hash = try digest(allocator, bytes);
    defer allocator.free(hash);
    if (!equal(hash, identity.sha256)) return reject(code, "file SHA256 differs from declared bytes");
}

fn verify_payload(io: std.Io, allocator: std.mem.Allocator, root: []const u8, file: PayloadFile, code: RefusalCode) ![:0]const u8 {
    const path = try payload_path(io, allocator, root, file.path);
    errdefer allocator.free(path);
    const bytes = zg.utils.mmap_file(path) catch return reject(.MissingInput, "cannot read payload file");
    defer zg.utils.munmap(bytes);
    try verify_bytes(allocator, bytes, file.identity(), code);
    return path;
}

fn promote_bf16(bytes: []const u8, out: []u8) !void {
    if (bytes.len % 2 != 0 or out.len != bytes.len * 2) return reject(.InvalidTensor, "BF16 byte extent mismatch");
    for (0..bytes.len / 2) |i| {
        const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16;
        std.mem.writeInt(u32, out[i * 4 ..][0..4], bits, .little);
    }
}

pub fn load_input_manifest(io: std.Io, allocator: std.mem.Allocator, args: ObservationArgs, weights: ?[]const u8) !CheckedInput {
    const bytes = zg.utils.mmap_file(args.input_manifest) catch return reject(.MissingInput, "input manifest is missing or inaccessible");
    defer zg.utils.munmap(bytes);
    if (bytes.len > 1024 * 1024) return reject(.InvalidManifest, "input manifest exceeds 1 MiB");
    const parsed = try parse_manifest(allocator, bytes, args.input_manifest_sha256);
    errdefer parsed.deinit();
    const m = parsed.value;
    const manifest_path = try std.Io.Dir.cwd().realPathFileAlloc(io, args.input_manifest, allocator);
    defer allocator.free(manifest_path);
    const root = std.fs.path.dirname(manifest_path) orelse return reject(.InvalidPath, "manifest directory missing");
    const descriptor_path = try payload_path(io, allocator, root, m.descriptor.path);
    defer allocator.free(descriptor_path);
    const descriptor_bytes = try zg.utils.mmap_file(descriptor_path);
    defer zg.utils.munmap(descriptor_bytes);
    try verify_bytes(allocator, descriptor_bytes, m.descriptor.identity(), .DescriptorMismatch);
    const descriptor = std.json.parseFromSlice(Descriptor, allocator, descriptor_bytes, .{}) catch return reject(.DescriptorMismatch, "descriptor JSON malformed");
    defer descriptor.deinit();
    try validate_descriptor(m, descriptor.value);
    const checkpoint = try payload_path(io, allocator, root, m.model.checkpoint.path);
    errdefer allocator.free(checkpoint);
    const checkpoint_bytes = zg.utils.mmap_file(checkpoint) catch return reject(.MissingInput, "checkpoint cannot be mapped");
    errdefer zg.utils.munmap(checkpoint_bytes);
    try verify_bytes(allocator, checkpoint_bytes, m.model.checkpoint.identity(), .CheckpointMismatch);
    if (weights) |supplied| {
        const resolved = std.Io.Dir.cwd().realPathFileAlloc(io, supplied, allocator) catch return reject(.CheckpointMismatch, "--weights cannot be resolved");
        defer allocator.free(resolved);
        if (!equal(resolved, checkpoint)) return reject(.CheckpointMismatch, "--weights differs from checked checkpoint path");
    }
    const config = try verify_payload(io, allocator, root, m.model.config, .ConfigMismatch);
    defer allocator.free(config);
    var tensors: [6][]const u8 = undefined;
    var observed: [6]ObservedTensor = undefined;
    var initialized: usize = 0;
    errdefer for (0..initialized) |i| {
        allocator.free(tensors[i]);
        allocator.free(observed[i].consumed.sha256);
    };
    for (m.tensors, 0..) |t, i| {
        const path = try payload_path(io, allocator, root, t.path);
        defer allocator.free(path);
        const raw = zg.utils.mmap_file(path) catch return reject(.MissingInput, "tensor payload missing");
        defer zg.utils.munmap(raw);
        try verify_bytes(allocator, raw, .{ .byte_length = t.byte_length, .sha256 = t.sha256 }, .TensorHashMismatch);
        const consumed = try allocator.alloc(u8, if (t.dtype == .bf16) raw.len * 2 else raw.len);
        errdefer allocator.free(consumed);
        if (t.dtype == .bf16) try promote_bf16(raw, consumed) else @memcpy(consumed, raw);
        const hash = try digest(allocator, consumed);
        tensors[i] = consumed;
        observed[i] = .{
            .name = t.name,
            .shape = t.shape,
            .encoding = t.encoding,
            .raw = .{ .dtype = t.dtype, .byte_length = t.byte_length, .sha256 = t.sha256 },
            .consumed = .{ .dtype = if (t.dtype == .i32) .i32 else .f32, .byte_length = consumed.len, .sha256 = hash },
        };
        initialized += 1;
    }
    return .{ .parsed = parsed, .manifest_sha256 = try allocator.dupe(u8, args.input_manifest_sha256), .checkpoint_path = checkpoint, .checkpoint_bytes = checkpoint_bytes, .tensors = tensors, .observed = observed, .allocator = allocator };
}

fn validate_descriptor(m: InputManifest, d: Descriptor) !void {
    if (d.format_version != 1 or !equal(d.batch.revision, m.workload.batch_revision) or d.batch.batch_size != m.workload.batch_size or d.batch.sequence_length != m.workload.sequence_length or
        d.checkpoint.byte_length != m.model.checkpoint.byte_length or !equal(d.checkpoint.sha256, m.model.checkpoint.sha256) or
        d.model_config.byte_length != m.model.config.byte_length or !equal(d.model_config.sha256, m.model.config.sha256))
        return reject(.DescriptorMismatch, "descriptor model/workload differs from manifest");
    for (m.tensors) |t| {
        var matches: usize = 0;
        for (d.batch.tensors) |dt| if (dt.name == t.name) {
            matches += 1;
            if (dt.dtype != t.dtype or dt.byte_length != t.byte_length or !equal(dt.sha256, t.sha256) or !std.mem.eql(i64, dt.shape, t.shape))
                return reject(.DescriptorMismatch, "descriptor tensor identity differs from manifest");
        };
        if (matches != 1) return reject(.DescriptorMismatch, "descriptor tensor set differs from manifest");
    }
}

fn fill_checked_batch(batch: Batch, input: CheckedInput) !void {
    for (input.observed, input.tensors) |t, bytes| {
        const tensor = switch (t.name) {
            .tokens => batch.x,
            .targets => batch.target_ids,
            .attention_mask => batch.attention_mask,
            .causal_mask => batch.mask,
            .sin => batch.sin,
            .cos => batch.cos,
        };
        const out = tensor.host_data_mut();
        if (out.len != bytes.len) return reject(.InvalidTensor, "consumed tensor extent differs from batch");
        for (0..bytes.len / 4) |i| {
            const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
            std.mem.writeInt(u32, out[i * 4 ..][0..4], bits, @import("builtin").cpu.arch.endian());
        }
    }
}

const ParameterMapping = struct { name: []const u8, shape: []const i64, buffer_index: usize, host: Tensor };

fn mapping_less(_: void, a: ParameterMapping, b: ParameterMapping) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn build_parameter_mapping(allocator: std.mem.Allocator, tree: *const zg.utils.Tree(Tensor), st: *const zg.SafetensorsFile) ![]ParameterMapping {
    if (tree.leaves.len != checked_workload.parameter_count) return reject(.InvalidObservation, "incomplete parameter tree");
    const mapping = try allocator.alloc(ParameterMapping, tree.leaves.len);
    errdefer allocator.free(mapping);
    for (tree.paths, tree.leaves, 0..) |name, tensor, i| {
        const view = st.get(name) catch return reject(.CheckpointMismatch, "parameter name missing from checkpoint");
        if (tensor.dtype != .f32 or tensor.dims().len != view.info.shape.len or try shape_elements(tensor.dims()) < checked_workload.sample_extent)
            return reject(.CheckpointMismatch, "parameter dtype/shape/sample extent mismatch");
        for (tensor.dims(), view.info.shape) |dim, checkpoint_dim| {
            if (dim != checkpoint_dim) return reject(.CheckpointMismatch, "checkpoint parameter shape mismatch");
        }
        mapping[i] = .{ .name = name, .shape = tree.leaves[i].dims(), .buffer_index = i, .host = tensor };
    }
    std.mem.sort(ParameterMapping, mapping, {}, mapping_less);
    for (mapping[1..], mapping[0 .. mapping.len - 1]) |a, b| {
        if (!mapping_less({}, b, a)) return reject(.InvalidObservation, "duplicate checkpoint parameter name");
    }
    return mapping;
}

fn encode_f32(values: []const f32, bytes: []u8) void {
    std.debug.assert(bytes.len == values.len * 4);
    for (values, 0..) |v, i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(v), .little);
}

fn collect_parameter_samples(allocator: std.mem.Allocator, executor: *zg.Executor, buffers: []const zg.Executor.Buffer, mapping: []const ParameterMapping, update_count: u32) ![]Sample {
    const samples = try allocator.alloc(Sample, mapping.len);
    errdefer allocator.free(samples);
    var initialized: usize = 0;
    errdefer for (samples[0..initialized]) |sample| allocator.free(sample.little_endian_f32_sha256);
    for (mapping, samples) |p, *sample| {
        if (p.buffer_index >= buffers.len) return reject(.InvalidObservation, "parameter buffer mapping out of bounds");
        var values: [checked_workload.sample_extent]f32 = undefined;
        const event = executor.download(buffers[p.buffer_index], std.mem.asBytes(&values)) catch
            return reject(.UnsupportedSampleReadback, "CPU parameter prefix download failed");
        if (event) |completion| {
            defer executor.release_event(completion);
            executor.wait(completion) catch return reject(.UnsupportedSampleReadback, "CPU parameter prefix completion failed");
        }
        for (values) |v| if (!std.math.isFinite(v)) return reject(.InvalidObservation, "parameter sample is nonfinite");
        var little_endian: [checked_workload.sample_extent * 4]u8 = undefined;
        encode_f32(&values, &little_endian);
        if (update_count == 0) {
            var host_values: [checked_workload.sample_extent]f32 = undefined;
            @memcpy(std.mem.asBytes(&host_values), p.host.host_data()[0..@sizeOf(@TypeOf(host_values))]);
            var expected: [@sizeOf(@TypeOf(host_values))]u8 = undefined;
            encode_f32(&host_values, &expected);
            if (!std.mem.eql(u8, &little_endian, &expected)) return reject(.CheckpointMismatch, "uploaded state0 prefix differs from converted checkpoint");
        }
        sample.* = .{ .name = p.name, .shape = p.shape, .update_count = update_count, .values_f32 = values, .little_endian_f32_sha256 = try digest(allocator, &little_endian) };
        initialized += 1;
    }
    return samples;
}

fn free_samples(allocator: std.mem.Allocator, samples: []Sample) void {
    for (samples) |sample| allocator.free(sample.little_endian_f32_sha256);
    allocator.free(samples);
}

fn ordered_names_digest(allocator: std.mem.Allocator, mapping: []const ParameterMapping) ![]const u8 {
    const names = try allocator.alloc([]const u8, mapping.len);
    defer allocator.free(names);
    for (mapping, names) |p, *name| name.* = p.name;
    const bytes = try std.json.Stringify.valueAlloc(allocator, names, .{});
    defer allocator.free(bytes);
    return digest(allocator, bytes);
}

const ObservedUpdates = struct {
    losses: Losses,
    untimed: [checked_workload.untimed_updates]UpdateTiming,
    timed: UpdateTiming,
};

fn observe_step(state: *zg.train.TrainState, timer: *zg.utils.LoopTimer, ordinal: u32) !struct { loss: Loss, timing: UpdateTiming } {
    try timer.start_step();
    var result = try state.step();
    errdefer result.loss.deinit();
    errdefer if (result.event) |completion| state.executor.release_event(completion);
    timer.mark("dispatch");
    if (result.event) |completion| try state.executor.wait(completion);
    const loss = try result.loss.item(f32);
    timer.mark("sync+read");
    if (!std.math.isFinite(loss)) return reject(.InvalidObservation, "loss is nonfinite");
    result.loss.deinit();
    if (result.event) |completion| state.executor.release_event(completion);
    timer.mark("cleanup");
    const dispatch = timer.phases[0].ns;
    const sync = timer.phases[1].ns;
    const cleanup = timer.phases[2].ns;
    timer.end_step(null);
    return .{
        .loss = .{ .value_f32 = loss, .producer_ordinal = ordinal, .evaluated_parameter_update_count = ordinal - 1 },
        .timing = .{
            .ordinal = ordinal,
            .input_update_count = ordinal - 1,
            .output_update_count = ordinal,
            .loss_evaluated_update_count = ordinal - 1,
            .dispatch_ns = dispatch,
            .scalar_read_sync_ns = sync,
            .cleanup_ns = cleanup,
            .step_through_scalar_read_ns = dispatch + sync,
            .full_step_ns = dispatch + sync + cleanup,
        },
    };
}

fn observe_updates(state: *zg.train.TrainState, timer: *zg.utils.LoopTimer, capture: anytype) !ObservedUpdates {
    try capture.collect(0);
    var updates: ObservedUpdates = undefined;
    for (&updates.untimed, 0..) |*timing, i| {
        const step = try observe_step(state, timer, @intCast(i + 1));
        timing.* = step.timing;
        if (i == 0) {
            updates.losses.initial = step.loss;
            try capture.collect(1);
        } else if (i == 1) updates.losses.after_step = step.loss;
    }
    updates.timed = (try observe_step(state, timer, @intCast(updates.untimed.len + 1))).timing;
    return updates;
}

const SampleCapture = struct {
    allocator: std.mem.Allocator,
    state: *zg.train.TrainState,
    mapping: []const ParameterMapping,
    initial: ?[]Sample = null,
    updated: ?[]Sample = null,

    fn collect(self: *SampleCapture, update_count: u32) !void {
        const samples = try collect_parameter_samples(self.allocator, self.state.executor, self.state.input_buffers, self.mapping, update_count);
        switch (update_count) {
            0 => self.initial = samples,
            1 => self.updated = samples,
            else => unreachable,
        }
    }

    fn deinit(self: *SampleCapture) void {
        if (self.initial) |samples| free_samples(self.allocator, samples);
        if (self.updated) |samples| free_samples(self.allocator, samples);
    }
};

fn validate_update(t: UpdateTiming, ordinal: u32) !void {
    const through = std.math.add(u64, t.dispatch_ns, t.scalar_read_sync_ns) catch return reject(.InvalidObservation, "step timing overflows");
    const full = std.math.add(u64, through, t.cleanup_ns) catch return reject(.InvalidObservation, "full timing overflows");
    if (t.ordinal != ordinal or t.input_update_count != ordinal - 1 or t.output_update_count != ordinal or t.loss_evaluated_update_count != ordinal - 1 or
        t.step_through_scalar_read_ns != through or t.full_step_ns != full or !equal(t.cleanup_scope, "loss_tensor_deinit_and_completion_event_release"))
        return reject(.InvalidObservation, "update state/timing mismatch");
}

fn validate_observation(allocator: std.mem.Allocator, success: Success) !void {
    if (success.losses.initial.producer_ordinal != 1 or success.losses.initial.evaluated_parameter_update_count != 0 or
        success.losses.after_step.producer_ordinal != 2 or success.losses.after_step.evaluated_parameter_update_count != 1 or
        !std.math.isFinite(success.losses.initial.value_f32) or !std.math.isFinite(success.losses.after_step.value_f32))
        return reject(.InvalidObservation, "loss producer/state mismatch or nonfinite loss");
    if (!equal(success.training.learning_rate_f32_bits, checked_workload.learning_rate_f32_bits) or success.training.sample_extent != checked_workload.sample_extent or
        success.training.framework_additional_warmup_steps != checked_workload.framework_additional_warmup_steps)
        return reject(.InvalidObservation, "observed training policy mismatch");
    const samples = success.parameter_samples;
    if (samples.initial.len != checked_workload.parameter_count or samples.updated.len != samples.initial.len)
        return reject(.InvalidObservation, "incomplete parameter samples");
    const names = try allocator.alloc([]const u8, samples.initial.len);
    defer allocator.free(names);
    for (samples.initial, samples.updated, 0..) |initial, updated, i| {
        if (!equal(initial.name, checked_workload.parameter_names[i]) or !equal(initial.name, updated.name) or !std.mem.eql(i64, initial.shape, updated.shape) or
            initial.update_count != 0 or updated.update_count != 1 or try shape_elements(initial.shape) < checked_workload.sample_extent or
            (i > 0 and !std.mem.lessThan(u8, samples.initial[i - 1].name, initial.name)))
            return reject(.InvalidObservation, "parameter ordering/shape/state mismatch");
        names[i] = initial.name;
        for ([_]Sample{ initial, updated }) |sample| {
            for (sample.values_f32) |v| if (!std.math.isFinite(v)) return reject(.InvalidObservation, "nonfinite parameter sample");
            var bytes: [checked_workload.sample_extent * 4]u8 = undefined;
            encode_f32(&sample.values_f32, &bytes);
            try verify_bytes(allocator, &bytes, .{ .byte_length = bytes.len, .sha256 = sample.little_endian_f32_sha256 }, .InvalidObservation);
        }
    }
    const name_bytes = try std.json.Stringify.valueAlloc(allocator, names, .{});
    defer allocator.free(name_bytes);
    try verify_bytes(allocator, name_bytes, .{ .byte_length = name_bytes.len, .sha256 = samples.ordered_names_sha256 }, .InvalidObservation);
    for (success.timing.untimed_updates, 0..) |t, i| try validate_update(t, @intCast(i + 1));
    try validate_update(success.timing.timed_update, @intCast(success.timing.untimed_updates.len + 1));
    if (success.timing.first_call_elapsed_ns != success.timing.untimed_updates[0].step_through_scalar_read_ns)
        return reject(.InvalidObservation, "first-call timing mismatch");
}

/// Publish complete JSON with an atomic rename that preserves existing evidence.
pub fn write_observation(io: std.Io, allocator: std.mem.Allocator, path: []const u8, observation: Observation) !void {
    if (observation.result == .success) try validate_observation(allocator, observation.result.success);
    const bytes = try std.json.Stringify.valueAlloc(allocator, observation, .{});
    defer allocator.free(bytes);
    const temporary = try std.fmt.allocPrint(allocator, "{s}.tmp", .{path});
    defer allocator.free(temporary);
    const dir = std.Io.Dir.cwd();
    var file = dir.createFile(io, temporary, .{ .exclusive = true }) catch return reject(.OutputFailed, "cannot reserve temporary observation file");
    defer dir.deleteFile(io, temporary) catch {};
    {
        defer file.close(io);
        file.writeStreamingAll(io, bytes) catch return reject(.OutputFailed, "cannot write observation JSON");
        file.sync(io) catch return reject(.OutputFailed, "cannot synchronize observation file");
    }
    dir.renamePreserve(temporary, dir, path, io) catch return reject(.OutputFailed, "cannot publish observation or output already exists");
}

/// Validated input, actual CPU identity, and publication destination.
pub const ObservationContext = struct {
    input: *const CheckedInput,
    input_readiness: struct {
        context: *anyopaque,
        wait: *const fn (*anyopaque, []const zg.Executor.Buffer) anyerror!void,
    },
    device: ObservedDevice,
    output: []const u8,
};

test validate_input_manifest {
    const allocator = std.testing.allocator;
    const bytes =
        \\{"schema":"zigrad.resident-llama-input","schema_version":1,"caller_contract_sha256":"16edcab5cfd2a1c7dcf4d40722b7d9b2cbce771c96dc160a19e59047e7f4a884","performance_specification":{"id":"spruce-meridian-cedar","revision":"1a10e7335108c55bb858c8fb13d08b9a9414d1d0","requirements":["OD-010","QR-003"]},"precision":"f32","device":{"kind":"cpu","backend":"pjrt"},"workload":{"batch_revision":"zg/llama-3.2-1b-batch1-seq4/v1","batch_size":1,"sequence_length":4},"descriptor":{"path":"llama_case.json","byte_length":2750,"sha256":"eff0806d3f8024daa2790bc507e8668406dcfed38b2b3aee54419e3156770289"},"model":{"checkpoint":{"path":"model.safetensors","byte_length":2471645608,"sha256":"1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"},"config":{"path":"config.json","byte_length":877,"sha256":"2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"},"tied_embedding_output_head":true},"tensors":[{"name":"tokens","dtype":"i32","shape":[1,4],"encoding":"raw_le_row_major","path":"inputs/tokens.bin","byte_length":16,"sha256":"8795b769791707562183172268e15e3efe31a30823635ca23f595ff2cbf47c82"},{"name":"targets","dtype":"i32","shape":[1,4],"encoding":"raw_le_row_major","path":"inputs/targets.bin","byte_length":16,"sha256":"64b95e5ff7a3a864cc626beb97a4069a5cb3975e91ea272f3ef7f3d42849fb47"},{"name":"attention_mask","dtype":"bf16","shape":[1,4],"encoding":"raw_le_row_major","path":"inputs/attention_mask.bin","byte_length":8,"sha256":"19c73878efaf4541a616d78b20071dd587d83591c7bd60bf150eb99a0136ea18"},{"name":"causal_mask","dtype":"bf16","shape":[4,4],"encoding":"raw_le_row_major","path":"inputs/causal_mask.bin","byte_length":32,"sha256":"52489d2995d5008a0a013dcced241b44085461ba31b2c8740ed5e5bab81784a5"},{"name":"sin","dtype":"bf16","shape":[4,32],"encoding":"raw_le_row_major","path":"inputs/sin.bin","byte_length":256,"sha256":"a880b9ab3c9ef40e63f649f0fe957486af0ce20ac0a237fa7d46f0f1da84f74c"},{"name":"cos","dtype":"bf16","shape":[4,32],"encoding":"raw_le_row_major","path":"inputs/cos.bin","byte_length":256,"sha256":"73eef29aec22ef302b406523e3330a712b084c745acc3f0fbbb1035e10fe299c"}],"training":{"learning_rate_f32_bits":"0x38d1b717","sample_extent":16,"untimed_updates_before_timed":2,"timed_update_ordinal":3},"expected":{"resident_revision":"b2ab033817e6757ce6a98dc0f9c7e4ec95bbc740","benchmark_revision":"5b6a108f5cf462aba335e958c7ce0a51990e4d2c","package_attribute":"zigrad-example-llama-training-cpu","configuration_key":"xla:cpu","source_files":[{"path":"examples/llama-training/src/main.zig","sha256":"e0a31196768a5c78ed13421fe24c53e9f0fe9d5c941fd4742266c41862b2b126"},{"path":"examples/llama-training/src/training.zig","sha256":"e103f0836cdbd15824a5ad37fef563ade2d5e3d5a03cd8e6b1abbc3ffc7e12f9"},{"path":"src/utils/loop_timer.zig","sha256":"7c3ea3f92a4be97f3bbbddf23de06d2f7e28fe367d6e8b3d441bfeca32f835bb"},{"path":"nix/flake/examples.nix","sha256":"184ab00b9782fb29ac90fb80a54983d73b6db176a9fd9c8836c7664e9c3018ad"},{"path":"nix/flake/configuration-definitions.nix","sha256":"5360f9129ace4f3c7df97cb6141ed2064488b73da1bccee32481b0f617522671"},{"path":"examples/llama-training/build.zig","sha256":"5c1230e92140102762c3cba3a8c5258aa9f6b9d325c3592c553e8e8c6a23301b"}]}}
    ;
    const hash = try digest(allocator, bytes);
    defer allocator.free(hash);
    const parsed = try parse_manifest(allocator, bytes, hash);
    defer parsed.deinit();
    var m = parsed.value;
    try validate_input_manifest(m);
    const descriptor_bytes =
        \\{"format_version":1,"program":{"source_repository":"https://github.com/Marco-Christiani/zigrad","source_revision":"46c581799e4cab505c94f5ac7e9480ff401f645d","artifact":{"byte_length":1237053,"sha256":"d457db271db880d97b51605fa927e928b50e510c9d6ac4f5129ffe02eaadcf55"}},"checkpoint":{"media_type":"application/x-safetensors","byte_length":2471645608,"sha256":"1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"},"model_config":{"media_type":"application/json","byte_length":877,"sha256":"2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"},"semantics":{"rms_norm_epsilon":1e-06,"learning_rate":0.0001,"attention_softmax_accumulator":"f32","loss_accumulator":"f32","loss_exponential_dtype":"bf16","loss_reduction":"mean-active-tokens"},"batch":{"revision":"zg/llama-3.2-1b-batch1-seq4/v1","batch_size":1,"sequence_length":4,"encoding":"little-endian-raw","generation":{"tokens":[128000,128009,128001,128008],"targets":[128009,128001,128008,128001],"active_tokens":4,"causal_mask":"lower-triangular-inclusive","rope":{"head_dimension":64,"theta":500000,"scaling_factor":32,"original_context_length":8192,"low_frequency_factor":1,"high_frequency_factor":4}},"tensors":[{"name":"tokens","dtype":"i32","shape":[1,4],"byte_length":16,"sha256":"8795b769791707562183172268e15e3efe31a30823635ca23f595ff2cbf47c82"},{"name":"targets","dtype":"i32","shape":[1,4],"byte_length":16,"sha256":"64b95e5ff7a3a864cc626beb97a4069a5cb3975e91ea272f3ef7f3d42849fb47"},{"name":"attention_mask","dtype":"bf16","shape":[1,4],"byte_length":8,"sha256":"19c73878efaf4541a616d78b20071dd587d83591c7bd60bf150eb99a0136ea18"},{"name":"causal_mask","dtype":"bf16","shape":[4,4],"byte_length":32,"sha256":"52489d2995d5008a0a013dcced241b44085461ba31b2c8740ed5e5bab81784a5"},{"name":"sin","dtype":"bf16","shape":[4,32],"byte_length":256,"sha256":"a880b9ab3c9ef40e63f649f0fe957486af0ce20ac0a237fa7d46f0f1da84f74c"},{"name":"cos","dtype":"bf16","shape":[4,32],"byte_length":256,"sha256":"73eef29aec22ef302b406523e3330a712b084c745acc3f0fbbb1035e10fe299c"}]}}
    ;
    const descriptor = try std.json.parseFromSlice(Descriptor, allocator, descriptor_bytes, .{});
    defer descriptor.deinit();
    try validate_descriptor(m, descriptor.value);
    var wrong_descriptor = descriptor.value;
    wrong_descriptor.batch.tensors[0].sha256 = checked_workload.contract_sha256;
    try std.testing.expectError(error.DescriptorMismatch, validate_descriptor(m, wrong_descriptor));
    wrong_descriptor = descriptor.value;
    wrong_descriptor.batch.tensors[0].shape = &.{ 4, 1 };
    try std.testing.expectError(error.DescriptorMismatch, validate_descriptor(m, wrong_descriptor));
    wrong_descriptor = descriptor.value;
    wrong_descriptor.model_config.byte_length += 1;
    try std.testing.expectError(error.DescriptorMismatch, validate_descriptor(m, wrong_descriptor));
    m.tensors[0].shape = &.{ std.math.maxInt(i64), std.math.maxInt(i64) };
    try std.testing.expectError(error.InvalidTensor, validate_input_manifest(m));
    m = parsed.value;
    m.tensors[0].dtype = .bf16;
    try std.testing.expectError(error.InvalidTensor, validate_input_manifest(m));
    m = parsed.value;
    m.tensors[1].name = m.tensors[0].name;
    try std.testing.expectError(error.InvalidTensor, validate_input_manifest(m));
    m = parsed.value;
    m.tensors[0].sha256 = "BAD";
    try std.testing.expectError(error.InvalidManifest, validate_input_manifest(m));
    m = parsed.value;
    m.training.learning_rate_f32_bits = "0x3f800000";
    try std.testing.expectError(error.InvalidManifest, validate_input_manifest(m));
    try std.testing.expectError(error.InvalidPath, validate_path("../inputs/tokens.bin"));
    try std.testing.expectError(error.InvalidPath, validate_path("inputs//tokens.bin"));
    try std.testing.expectError(error.ManifestHashMismatch, parse_manifest(allocator, bytes[0 .. bytes.len - 1], hash));
    const malformed = "{\"schema\":\"x\",\"schema\":\"duplicate\"}";
    const bad_hash = try digest(allocator, malformed);
    defer allocator.free(bad_hash);
    try std.testing.expectError(error.InvalidManifest, parse_manifest(allocator, malformed, bad_hash));
    const unknown = try std.fmt.allocPrint(allocator, "{{\"extra\":true,{s}", .{bytes[1..]});
    defer allocator.free(unknown);
    const unknown_hash = try digest(allocator, unknown);
    defer allocator.free(unknown_hash);
    try std.testing.expectError(error.InvalidManifest, parse_manifest(allocator, unknown, unknown_hash));
}

test load_input_manifest {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const missing = try std.fs.path.join(allocator, &.{ root, "missing.json" });
    defer allocator.free(missing);
    try std.testing.expectError(error.MissingInput, load_input_manifest(io, allocator, .{
        .input_manifest = missing,
        .input_manifest_sha256 = checked_workload.contract_sha256,
        .output = "unused.json",
    }, null));
    try temporary.dir.writeFile(io, .{ .sub_path = "short.bin", .data = "abc" });
    const hash = try digest(allocator, "abcd");
    defer allocator.free(hash);
    try std.testing.expectError(error.TensorHashMismatch, verify_payload(io, allocator, root, .{ .path = "short.bin", .byte_length = 4, .sha256 = hash }, .TensorHashMismatch));
    try std.testing.expectError(error.DescriptorMismatch, verify_bytes(allocator, "abce", .{ .byte_length = 4, .sha256 = hash }, .DescriptorMismatch));
    const refused = Observation{ .result = .{ .refused = .{ .code = .MissingInput, .message = "fixture missing input" } } };
    const output = try std.fs.path.join(allocator, &.{ root, "observation.json" });
    defer allocator.free(output);
    try write_observation(io, allocator, output, refused);
    try std.testing.expectError(error.OutputFailed, write_observation(io, allocator, output, refused));
    const raw = try zg.utils.mmap_file(output);
    defer zg.utils.munmap(raw);
    const decoded = try std.json.parseFromSlice(Observation, allocator, raw, .{});
    defer decoded.deinit();
    try std.testing.expectEqual(RefusalCode.MissingInput, decoded.value.result.refused.code);
}

test promote_bf16 {
    const allocator = std.testing.allocator;
    const raw = [_]u8{ 0x80, 0x3f, 0x80, 0xbf, 0x01, 0x00, 0x80, 0x7f };
    var out: [16]u8 = undefined;
    try promote_bf16(&raw, &out);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0x80, 0x3f, 0, 0, 0x80, 0xbf, 0, 0, 0x01, 0, 0, 0, 0x80, 0x7f }, &out);
    try std.testing.expectError(error.InvalidTensor, promote_bf16(raw[0..7], &out));
    // Each mask/rotary extent uses independently computed promoted-byte hashes.
    const cases = .{
        .{ .name = TensorName.attention_mask, .count = 4, .hash = "fd801740524b1e1c23a61b5400725b76428652827ba1261b83243eb5ce551402" },
        .{ .name = TensorName.causal_mask, .count = 16, .hash = "018f4ea8ef8d642393e6535bf9106cf3b489b90877cc11021017dd69f9b8e7ea" },
        .{ .name = TensorName.sin, .count = 128, .hash = "3746552b62ac1dd063a110871d7c91973fe0222908d3b26e4789773f97e0843e" },
        .{ .name = TensorName.cos, .count = 128, .hash = "3746552b62ac1dd063a110871d7c91973fe0222908d3b26e4789773f97e0843e" },
    };
    inline for (cases) |case| {
        var input: [case.count * 2]u8 = undefined;
        var promoted: [case.count * 4]u8 = undefined;
        for (0..case.count) |i| std.mem.writeInt(u16, input[i * 2 ..][0..2], if (i % 2 == 0) 0x3f80 else 0xbf80, .little);
        try promote_bf16(&input, &promoted);
        const hash = try digest(allocator, &promoted);
        defer allocator.free(hash);
        try std.testing.expectEqualStrings(case.hash, hash);
        const shape = tensor_shape(case.name, checked_workload.batch_size, checked_workload.sequence_length);
        try std.testing.expectEqual(@as(u64, case.count), try shape_elements(&shape));
    }
}

test observe_updates {
    const Fake = struct {
        interface: zg.Executor = .{ .device = .{ .platform = .cpu }, .vtable = &vtable },
        parameters: [2][checked_workload.sample_extent]f32 = .{ @splat(1), @splat(2) },
        next: [2][checked_workload.sample_extent]f32 = undefined,
        loss: f32 = 0,
        update_count: u32 = 0,
        waited: bool = true,
        events_waited: usize = 0,
        events_released: usize = 0,
        buffers_released: usize = 0,
        const vtable: zg.Executor.VTable = .{
            .upload = upload,
            .download = download,
            .invoke = invoke,
            .await_event = wait,
            .release_buffer = release,
            .release_event = release_event,
            .release_program = release_program,
        };
        fn parent(executor: *zg.Executor) *@This() {
            return @fieldParentPtr("interface", executor);
        }
        fn upload(_: *zg.Executor, _: []const u8, _: zg.DType, _: []const i64) zg.Executor.Error!zg.Executor.Buffer {
            return error.Unsupported;
        }
        fn download(executor: *zg.Executor, buffer: zg.Executor.Buffer, destination: []u8) zg.Executor.Error!?zg.Executor.Event {
            const self = parent(executor);
            if (!self.waited) return error.InvalidArgument;
            const source: [*]const u8 = @ptrCast(buffer.handle);
            @memcpy(destination, source[0..destination.len]);
            return null;
        }
        fn invoke(executor: *zg.Executor, _: *anyopaque, inputs: []const zg.Executor.Buffer, outputs: []zg.Executor.Buffer, options: zg.Executor.InvokeOptions) zg.Executor.Error!?zg.Executor.Event {
            const self = parent(executor);
            if (options.donated_input_indices.len != 2) return error.InvalidArgument;
            self.loss = @floatFromInt(self.update_count);
            for (inputs, outputs[1..], 0..) |input, *output, i| {
                const source: *[checked_workload.sample_extent]f32 = @ptrCast(@alignCast(input.handle));
                const destination = if (input.handle == @as(*anyopaque, @ptrCast(&self.parameters[i]))) &self.next[i] else &self.parameters[i];
                for (source, destination) |v, *out| out.* = v + 10;
                output.* = .{ .handle = @ptrCast(destination) };
            }
            outputs[0] = .{ .handle = @ptrCast(&self.loss) };
            self.update_count += 1;
            self.waited = false;
            return .{ .handle = @ptrCast(self) };
        }
        fn wait(executor: *zg.Executor, _: zg.Executor.Event) zg.Executor.Error!void {
            const self = parent(executor);
            self.waited = true;
            self.events_waited += 1;
        }
        fn release(executor: *zg.Executor, _: zg.Executor.Buffer) void {
            parent(executor).buffers_released += 1;
        }
        fn release_event(executor: *zg.Executor, _: zg.Executor.Event) void {
            parent(executor).events_released += 1;
        }
        fn release_program(_: *zg.Executor, _: *anyopaque) void {}
    };
    const allocator = std.testing.allocator;
    var fake = Fake{};
    const tensors = [_]Tensor{
        Tensor.from_buffer(&fake.interface, .{ .handle = @ptrCast(&fake.parameters[0]) }, .f32, &.{checked_workload.sample_extent}),
        Tensor.from_buffer(&fake.interface, .{ .handle = @ptrCast(&fake.parameters[1]) }, .f32, &.{checked_workload.sample_extent}),
    };
    var host0 = try Tensor.host(.f32, &.{checked_workload.sample_extent}, .{ .alloc = allocator });
    defer host0.deinit();
    var host1 = try Tensor.host(.f32, &.{checked_workload.sample_extent}, .{ .alloc = allocator });
    defer host1.deinit();
    @memset(host0.as_slice(f32), 1);
    @memset(host1.as_slice(f32), 2);
    var mapping = [_]ParameterMapping{
        .{ .name = "model.layers.2.mlp.up_proj.weight", .shape = &.{checked_workload.sample_extent}, .buffer_index = 0, .host = host0 },
        .{ .name = "model.layers.10.mlp.up_proj.weight", .shape = &.{checked_workload.sample_extent}, .buffer_index = 1, .host = host1 },
    };
    std.mem.sort(ParameterMapping, &mapping, {}, mapping_less);
    try std.testing.expectEqual(@as(usize, 1), mapping[0].buffer_index);
    var state = try zg.train.TrainState.init(allocator, .{ .executor = &fake.interface, .handle = @ptrCast(&fake) }, &tensors, 3, .{ .donated_input_indices = &.{ 0, 1 } });
    defer state.deinit(.all);
    var capture = SampleCapture{ .allocator = allocator, .state = &state, .mapping = &mapping };
    defer capture.deinit();
    var timer = zg.utils.LoopTimer{ .io = std.testing.io, .label = "fake", .quiet = true };
    const updates = try observe_updates(&state, &timer, &capture);
    try std.testing.expectEqual(@as(f32, 0), updates.losses.initial.value_f32);
    try std.testing.expectEqual(@as(f32, 1), updates.losses.after_step.value_f32);
    try std.testing.expectEqual(@as(f32, 2), capture.initial.?[0].values_f32[0]);
    try std.testing.expectEqual(@as(f32, 12), capture.updated.?[0].values_f32[0]);
    try std.testing.expectEqual(@as(f32, 32), fake.next[1][0]);
    try std.testing.expectEqual(@as(usize, 3), fake.events_waited);
    try std.testing.expectEqual(@as(usize, 3), fake.events_released);
    try std.testing.expectEqual(@as(usize, 9), fake.buffers_released);
    try std.testing.expectEqual(@as(u32, 2), updates.timed.input_update_count);
    try std.testing.expectEqual(@as(u32, 3), updates.timed.output_update_count);
    try validate_update(updates.timed, 3);
    var invalid = updates.timed;
    invalid.full_step_ns += 1;
    try std.testing.expectError(error.InvalidObservation, validate_update(invalid, 3));
    invalid = updates.timed;
    invalid.input_update_count = 1;
    try std.testing.expectError(error.InvalidObservation, validate_update(invalid, 3));
    @memset(host1.as_slice(f32), 99);
    try std.testing.expectError(error.CheckpointMismatch, collect_parameter_samples(allocator, &fake.interface, state.input_buffers, &mapping, 0));
}

test collect_parameter_samples {
    const allocator = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 146), checked_workload.parameter_names.len);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", checked_workload.parameter_names[0]);
    try std.testing.expectEqualStrings("model.norm.weight", checked_workload.parameter_names[checked_workload.parameter_names.len - 1]);
    for (checked_workload.parameter_names) |name| try std.testing.expect(!std.mem.startsWith(u8, name, "lm_head"));
    var mapping = [_]ParameterMapping{
        .{ .name = "model.layers.2.mlp.up_proj.weight", .shape = &.{16}, .buffer_index = 0, .host = undefined },
        .{ .name = "model.layers.10.mlp.up_proj.weight", .shape = &.{16}, .buffer_index = 1, .host = undefined },
    };
    std.mem.sort(ParameterMapping, &mapping, {}, mapping_less);
    const hash = try ordered_names_digest(allocator, &mapping);
    defer allocator.free(hash);
    const expected = try digest(allocator, "[\"model.layers.10.mlp.up_proj.weight\",\"model.layers.2.mlp.up_proj.weight\"]");
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, hash);
    var host = try Tensor.host(.f32, &.{16}, .{ .alloc = allocator });
    defer host.deinit();
    @memset(host.as_slice(f32), 1);
    const leaves: [checked_workload.parameter_count]Tensor = @splat(host);
    var tree = try zg.utils.Tree(Tensor).from_slices(allocator, &leaves, &checked_workload.parameter_names);
    defer tree.deinit();
    const parameters = try tree.extract(LlamaParams);
    const serialized = try zg.to_safetensors(LlamaParams, parameters, allocator);
    defer allocator.free(serialized);
    var checkpoint = try zg.SafetensorsFile.deserialize(serialized, allocator);
    defer checkpoint.deinit();
    const checked_mapping = try build_parameter_mapping(allocator, &tree, &checkpoint);
    defer allocator.free(checked_mapping);
    try std.testing.expectEqual(checked_workload.parameter_count, checked_mapping.len);
    try std.testing.expectEqualSlices(i64, &.{16}, checked_mapping[0].shape);
    tree.leaves[0].shape = .from_slice(&.{32});
    try std.testing.expectError(error.CheckpointMismatch, build_parameter_mapping(allocator, &tree, &checkpoint));
    const values = [_]f32{ 0, -0.0, 1e-4, @bitCast(@as(u32, 0x00000001)), @bitCast(@as(u32, 0x7f7fffff)) };
    const bytes = try std.json.Stringify.valueAlloc(allocator, values, .{});
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(@TypeOf(values), allocator, bytes, .{});
    defer parsed.deinit();
    for (values, parsed.value) |a, b| try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
}

test validate_observation {
    const allocator = std.testing.allocator;
    const bytes: [checked_workload.sample_extent * 4]u8 = @splat(0);
    const sample_hash = try digest(allocator, &bytes);
    defer allocator.free(sample_hash);
    var initial: [checked_workload.parameter_count]Sample = undefined;
    var updated: [checked_workload.parameter_count]Sample = undefined;
    for (checked_workload.parameter_names, &initial, &updated) |name, *a, *b| {
        a.* = .{ .name = name, .shape = &.{16}, .update_count = 0, .values_f32 = @splat(0), .little_endian_f32_sha256 = sample_hash };
        b.* = a.*;
        b.update_count = 1;
    }
    const names_json = try std.json.Stringify.valueAlloc(allocator, checked_workload.parameter_names, .{});
    defer allocator.free(names_json);
    const names_hash = try digest(allocator, names_json);
    defer allocator.free(names_hash);
    var timings: [3]UpdateTiming = undefined;
    for (&timings, 0..) |*t, i| t.* = .{
        .ordinal = @intCast(i + 1),
        .input_update_count = @intCast(i),
        .output_update_count = @intCast(i + 1),
        .loss_evaluated_update_count = @intCast(i),
        .dispatch_ns = 1,
        .scalar_read_sync_ns = 2,
        .cleanup_ns = 3,
        .step_through_scalar_read_ns = 3,
        .full_step_ns = 6,
    };
    var success = Success{
        .device = .{ .platform_name = "cpu", .device_kind = "synthetic", .ordinal = 0 },
        .identity = undefined,
        .losses = .{ .initial = .{ .value_f32 = 1, .producer_ordinal = 1, .evaluated_parameter_update_count = 0 }, .after_step = .{ .value_f32 = 0.5, .producer_ordinal = 2, .evaluated_parameter_update_count = 1 } },
        .parameter_samples = .{ .ordered_names_sha256 = names_hash, .initial = &initial, .updated = &updated },
        .timing = .{ .preparation = .{ .trace_elapsed_ns = 0, .pipeline_preparation_elapsed_ns = 0 }, .first_call_elapsed_ns = 3, .untimed_updates = timings[0..2].*, .timed_update = timings[2] },
    };
    try validate_observation(allocator, success);
    success.losses.after_step.producer_ordinal = 1;
    try std.testing.expectError(error.InvalidObservation, validate_observation(allocator, success));
    success.losses.after_step.producer_ordinal = 2;
    initial[0].values_f32[0] = std.math.nan(f32);
    try std.testing.expectError(error.InvalidObservation, validate_observation(allocator, success));
    initial[0].values_f32[0] = 0;
    initial[0].name = "lm_head.weight";
    try std.testing.expectError(error.InvalidObservation, validate_observation(allocator, success));
    initial[0].name = checked_workload.parameter_names[0];
    success.parameter_samples.updated = updated[0..1];
    try std.testing.expectError(error.InvalidObservation, validate_observation(allocator, success));
}
