const std = @import("std");

const Executor = @import("execution.zig");
const Tensor = @import("tensor.zig");
const TensorTree = @import("utils.zig").Tree(Tensor);

/// Manages device buffer state across training steps.
///
/// `TrainState` manages execution after tracing and explicit compilation.
///
/// After compiling a train step, `TrainState` handles the repetitive buffer
///  plumbing.
///
/// A train step is typically a function using `transforms.value_and_grad` and
///  an optimizer. Donatable inputs are replaced with their updated outputs,
///  while batch inputs remain borrowed from the caller. `step` invokes the
///  loaded program, extracts the loss buffer, and performs the swap.
///
/// ## Typical usage
///
/// ```zig
/// var traced = try zg.trace(train_step, allocator, specs, .{ .name = "train_step" });
/// defer traced.deinit();
/// var loaded_program = try pipeline.run(zg.Executor.LoadedProgram, &traced.program, &ctx);
/// defer loaded_program.deinit();
/// const entry = try traced.program.resolve_entry();
/// var state = try TrainState.init(
///     allocator,
///     loaded_program,
///     initial_tensors,
///     traced.program.get_function_by_id(entry).?.returns.len,
///     .{ .donated_input_indices = comptime donated_input_indices(@TypeOf(specs), &.{0}) },
/// );
/// defer state.deinit(.all);
/// for (0..num_steps) |_| {
///     state.set_batch(batch_tensors);
///     const result = try state.step();
///     const loss_val = try result.loss.item(f32);
///     result.loss.deinit();
/// }
/// ```
pub const TrainState = struct {
    input_buffers: []Executor.Buffer,
    output_buffers: []Executor.Buffer,
    loaded_program: Executor.LoadedProgram,
    executor: *Executor,
    /// Number of inputs managed by `TrainState` through donation.
    ///
    /// These occupy `input_buffers[0..donatable_count]`. Borrowed inputs occupy
    ///  `input_buffers[donatable_count..]`.
    donatable_count: usize,
    donated_input_indices: []usize,
    /// DType of the loss output (output[0]). Used to wrap the raw buffer
    ///  as a Tensor in `StepResult`.
    loss_dtype: @import("pr").DType,
    allocator: std.mem.Allocator,

    pub const StepResult = struct {
        loss: Tensor,
        event: ?Executor.Event,
    };

    /// Initialize from individual components.
    ///
    /// `initial_tensors` must be in the same flattened order as the spec tree
    ///  passed to `trace`. All tensors must be device-backed.
    ///
    /// Donation is specified via `donated_input_indices` in opts. Use the
    ///  `donated_input_indices` helper to flatten donated argument positions.
    /// Donatable inputs must precede non-donatable inputs in the flattened spec
    ///  order. This is the natural layout when params are the first argument and
    ///  batch data follows.
    ///
    /// `TrainState` releases donatable buffers during swaps and `deinit`.
    ///  The caller keeps borrowed non-donatable buffers and `loaded_program`
    ///  alive.
    pub const InitOpts = struct {
        /// Flat input positions available for buffer reuse.
        ///
        /// Indices must form a leading sequence starting at zero. Use
        ///  `donated_input_indices` to flatten argument positions from spec types.
        donated_input_indices: []const usize = &.{},
        /// DType of the loss output (output[0]). Defaults to f32.
        /// Override for mixed-precision training (e.g. bf16 loss with
        ///  f32 upcast).
        loss_dtype: @import("pr").DType = .f32,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        loaded_program: Executor.LoadedProgram,
        initial_tensors: []const Tensor,
        output_arity: usize,
        opts: InitOpts,
    ) (Executor.Error || error{ UnsupportedAval, InvalidDonation, InvalidOutputArity })!TrainState {
        const executor = loaded_program.executor;
        if (opts.donated_input_indices.len > initial_tensors.len) {
            return error.InvalidDonation;
        }
        const donatable_count = opts.donated_input_indices.len;
        if (output_arity < donatable_count + 1) return error.InvalidOutputArity;
        for (opts.donated_input_indices, 0..) |input_index, expected| {
            if (input_index != expected) return error.InvalidDonation;
        }

        const input_buffers = try allocator.alloc(Executor.Buffer, initial_tensors.len);
        errdefer allocator.free(input_buffers);
        for (input_buffers, initial_tensors) |*slot, tensor| {
            const device = switch (tensor.backing) {
                .device => |device| device,
                .traced, .host, .abstract => return error.UnsupportedAval,
            };
            if (device.executor != executor) return error.InvalidArgument;
            slot.* = device.buffer;
        }

        const output_buffers = try allocator.alloc(Executor.Buffer, output_arity);
        errdefer allocator.free(output_buffers);
        const donated_indices = try allocator.alloc(usize, donatable_count);
        errdefer allocator.free(donated_indices);
        @memcpy(donated_indices, opts.donated_input_indices);

        return .{
            .input_buffers = input_buffers,
            .output_buffers = output_buffers,
            .loaded_program = loaded_program,
            .executor = executor,
            .donatable_count = donatable_count,
            .donated_input_indices = donated_indices,
            .loss_dtype = opts.loss_dtype,
            .allocator = allocator,
        };
    }

    /// Execute one training step. Swaps donatable buffers in-place.
    ///
    /// Returns `StepResult` with the loss as a device Tensor and optional
    ///  completion event. Caller owns the loss tensor (call `deinit` or
    ///  use `item()` then `deinit`). Donated input buffers are replaced with
    ///  their corresponding outputs. Distinct old buffers are freed.
    pub fn step(self: *TrainState) Executor.Error!StepResult {
        const event = try self.executor.invoke(
            self.loaded_program,
            self.input_buffers,
            self.output_buffers,
            .{ .donated_input_indices = self.donated_input_indices },
        );

        // TODO: APIs have generalized so the packing assumptions made here may no longer hold.
        const loss_buf = self.output_buffers[0];

        // Swap donatable buffers: output[1+i] replaces input[i] for each
        //  donatable input. Output index is offset by 1 (loss is output[0]).
        for (self.input_buffers[0..self.donatable_count], 1..) |*old, out_idx| {
            const new_buf = self.output_buffers[out_idx];
            if (new_buf.handle != old.handle) {
                self.executor.release(old.*);
                old.* = new_buf;
            }
        }

        return .{
            .loss = Tensor.from_buffer(self.executor, loss_buf, self.loss_dtype, &.{}),
            .event = event,
        };
    }

    /// Replace a non-donatable (borrowed) input by flat index into the
    ///  non-donatable slice (not the full input array).
    ///
    /// The tensor must be device-backed. The old buffer is NOT freed
    ///  (caller owns non-donatable buffers).
    pub fn set_batch_buf(self: *TrainState, batch_idx: usize, tensor: Tensor) void {
        std.debug.assert(self.donatable_count + batch_idx < self.input_buffers.len);
        self.input_buffers[self.donatable_count + batch_idx] = tensor.buffer() catch
            @panic("set_batch_buf: tensor must be device-backed");
    }

    /// Replace all non-donatable inputs from a flat tensor slice.
    ///
    /// `batch_tensors` must have exactly as many elements as there are
    ///  non-donatable inputs. All must be device-backed.
    pub fn set_batch(self: *TrainState, batch_tensors: []const Tensor) void {
        const batch_bufs = self.input_buffers[self.donatable_count..];
        std.debug.assert(batch_tensors.len == batch_bufs.len);
        for (batch_bufs, batch_tensors) |*slot, t| {
            slot.* = t.buffer() catch
                @panic("set_batch: tensor must be device-backed");
        }
    }

    pub const DeinitScope = enum {
        /// Release only donatable parameter buffers.
        donatable,
        /// Free non-donatable (borrowed batch) buffers only.
        non_donatable,
        /// Free all input buffers regardless of donation status.
        all,
    };

    /// Release device buffers and free internal allocations.
    ///
    /// Buffer release follows `scope`.
    ///
    /// `.donatable` releases parameter buffers, `.non_donatable` releases
    ///  borrowed batch buffers, and `.all` releases both. Internal slice
    ///  allocations are always freed.
    ///
    /// After `deinit`, the TrainState must not be used.
    pub fn deinit(self: *TrainState, scope: DeinitScope) void {
        switch (scope) {
            .donatable => for (self.input_buffers[0..self.donatable_count]) |buf| self.executor.release(buf),
            .non_donatable => for (self.input_buffers[self.donatable_count..]) |buf| self.executor.release(buf),
            .all => for (self.input_buffers) |buf| self.executor.release(buf),
        }
        self.allocator.free(self.input_buffers);
        self.allocator.free(self.output_buffers);
        self.allocator.free(self.donated_input_indices);
    }

    /// Release device buffers without freeing internal allocations.
    ///
    /// Unlike `deinit`, the TrainState remains valid after this call. Only the
    ///  specified buffers are released (handles set to undefined). This enables
    ///  sequential calls: `release_buffers(.donatable)` then `release_buffers(.non_donatable)`.
    /// Follow with `deinit(.all)` to free internal slice allocations (no buffers
    ///  will be released since handles are already invalidated).
    pub fn release_buffers(self: *TrainState, scope: DeinitScope) void {
        const range = switch (scope) {
            .donatable => self.input_buffers[0..self.donatable_count],
            .non_donatable => self.input_buffers[self.donatable_count..],
            .all => self.input_buffers,
        };
        for (range) |*buf| {
            self.executor.release(buf.*);
            buf.* = undefined;
        }
    }
};

/// Flatten donated argument positions into input leaf indices.
///
/// `SpecsTuple` is the type of the spec tuple passed to `zg.trace` (e.g.
///  `@TypeOf(.{ params_spec, batch_spec })`). `donated_argnums` lists the
///  argument positions that are donatable (e.g. `&.{0}` means "first arg is
///  donated").
///
/// Returns indices suitable for `Executor.InvokeOptions` and
///  `TrainState.InitOpts`.
///
/// ```zig
/// const Specs = @TypeOf(.{ params_spec, batch_spec });
/// var state = try TrainState.init(allocator, loaded_program, tensors, output_arity, .{
///     .donated_input_indices = comptime donated_input_indices(Specs, &.{0}),
/// });
/// ```
pub fn donated_input_indices(
    comptime SpecsTuple: type,
    comptime donated_argnums: []const usize,
) []const usize {
    comptime {
        const info = @typeInfo(SpecsTuple);
        if (info != .@"struct" or !info.@"struct".is_tuple)
            @compileError("donated_input_indices: SpecsTuple must be a tuple type");

        const fields = info.@"struct";

        // Validate argument positions and count their flattened leaves.
        var count: usize = 0;
        for (donated_argnums, 0..) |arg_idx, position| {
            if (arg_idx >= fields.len) @compileError("donated argument position is out of bounds");
            if (is_donated(donated_argnums[0..position], arg_idx))
                @compileError("donated argument positions must be unique");
            count += TensorTree.leaf_count(fields[arg_idx].type);
        }

        var result: [count]usize = undefined;
        var out_idx: usize = 0;
        var flat_offset: usize = 0;
        for (fields, 0..) |field, arg_idx| {
            const n = TensorTree.leaf_count(field.type);
            if (is_donated(donated_argnums, arg_idx)) {
                for (0..n) |i| {
                    result[out_idx] = flat_offset + i;
                    out_idx += 1;
                }
            }
            flat_offset += n;
        }

        const final = result;
        return &final;
    }
}

fn is_donated(comptime donated: []const usize, comptime arg_idx: usize) bool {
    for (donated) |d| {
        if (d == arg_idx) return true;
    }
    return false;
}

test donated_input_indices {
    const Params = struct { weight: Tensor, bias: Tensor };
    const Specs = @Tuple(&.{ Params, Tensor, [2]Tensor });
    const donated = comptime donated_input_indices(Specs, &.{ 0, 2 });
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4 }, donated);
}
