//! Binding between traced functions and loaded programs.
//!
//! A traced callable exposes its PR program for explicit transformation,
//!  lowering, and compilation. Binding a loaded program restores the original
//!  function's structured input and output types.
//!
//! ## Usage
//!
//! ```zig
//! var traced = try zg.trace(
//!     predict,
//!     allocator,
//!     .{ params_spec, inputs_spec },
//!     .{ .name = "predict" },
//! );
//! defer traced.deinit();
//!
//! // Application composition supplies operations for the selected target.
//! var pipeline = zg.Pipeline.init(allocator);
//! defer pipeline.deinit();
//! try pipeline.add(validate);
//! try pipeline.add(lower_to_target);
//! try pipeline.add(terminal_backend);
//! const loaded_program = try pipeline.run(
//!     zg.Executor.LoadedProgram,
//!     &traced.program,
//!     &context,
//! );
//! var predict_fn = try traced.bind(loaded_program);
//! defer predict_fn.deinit();
//!
//! const inputs: @TypeOf(predict_fn).InputType = .{ params, images };
//! var predictions = try predict_fn.call(&inputs);
//! defer predictions.deinit();
//! ```
//!
const std = @import("std");

const pr = @import("pr");
const Executor = @import("execution.zig");
const Tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const meta = utils.meta;
const RuntimeOf = utils.RuntimeOf;
const TensorTree = utils.Tree(Tensor);

/// A PR program paired with the function that produced it.
pub fn Traced(
    comptime func: anytype,
    comptime SpecsTuple: type,
) type {
    return struct {
        const Self = @This();

        /// Traced PR program released by `deinit`.
        program: pr.Program,

        /// PR function associated with the call contract.
        function: pr.FunctionId,

        /// Bind a loaded program compiled from this traced program.
        ///
        /// On success, `Compiled.deinit` releases `loaded_program`. `Traced.deinit`
        ///  releases the PR program separately.
        pub fn bind(
            self: *const Self,
            loaded_program: Executor.LoadedProgram,
        ) error{ NoEntry, EntryMismatch }!Compiled(func, SpecsTuple) {
            if (self.program.entry != self.function) return error.EntryMismatch;
            const entry = self.program.get_function_by_id(self.function) orelse return error.NoEntry;
            return Compiled(func, SpecsTuple).init(loaded_program, entry);
        }

        /// Release the traced PR program.
        pub fn deinit(self: *Self) void {
            self.program.deinit();
            self.* = undefined;
        }
    };
}

/// Compiled function parameterized by its source call contract.
pub fn Compiled(
    comptime func: anytype,
    comptime SpecsTuple: type,
) type {
    const FullReturnType = DeriveReturnType(func);
    const CallReturn = RuntimeOf(FullReturnType);
    const input_count = TensorTree.leaf_count(SpecsTuple);
    const output_count = TensorTree.leaf_count(FullReturnType);

    return struct {
        const Self = @This();

        /// The structured input type accepted by `call()`.
        pub const InputType = RuntimeOf(SpecsTuple);

        /// Structured return type declared by the traced function.
        pub const ReturnType = CallReturn;

        /// Failures reported while validating inputs or invoking the loaded program.
        pub const Error = Executor.Error || error{UnsupportedAval};

        loaded_program: Executor.LoadedProgram,

        /// Output tensor descriptors used to reconstruct each call result.
        ///
        /// Buffer handles are overwritten after each invocation.
        output_tensors: [output_count]Tensor,

        fn init(
            loaded_program: Executor.LoadedProgram,
            entry: pr.Function,
        ) Self {
            std.debug.assert(entry.returns.len == output_count);
            const executor = loaded_program.executor;

            var output_tensors: [output_count]Tensor = undefined;
            for (entry.returns, &output_tensors) |ret_var, *output| {
                const t = ret_var.as_tensor();
                output.* = .{
                    .dtype = t.dtype,
                    .shape = .from_slice(t.shape.dims),
                    .backing = .{ .device = .{ .buffer = undefined, .executor = executor } },
                };
            }

            return .{
                .loaded_program = loaded_program,
                .output_tensors = output_tensors,
            };
        }

        /// Execute the compiled function with structured inputs.
        ///
        /// Caller owns the result.
        pub fn call(self: *Self, args: *const InputType) Error!CallReturn {
            var input_tensors: [input_count]Tensor = undefined;
            var flat_idx: usize = 0;
            meta.flatten(Tensor, InputType, args.*, &input_tensors, &flat_idx);
            std.debug.assert(flat_idx == input_count);

            var input_buffers: [input_count]Executor.Buffer = undefined;
            for (input_tensors, &input_buffers) |tensor, *buffer| {
                const device = switch (tensor.backing) {
                    .device => |device| device,
                    .traced, .host, .abstract => return error.UnsupportedAval,
                };
                if (device.executor != self.loaded_program.executor) return error.InvalidArgument;
                buffer.* = device.buffer;
            }

            var output_buffers: [output_count]Executor.Buffer = undefined;
            const executor = self.loaded_program.executor;
            const event = try executor.invoke(
                self.loaded_program,
                &input_buffers,
                &output_buffers,
                .{},
            );
            if (event) |completion| {
                defer executor.release_event(completion);
                try executor.wait(completion);
            }

            for (&self.output_tensors, output_buffers) |*tensor, buffer| {
                tensor.backing.device.buffer = buffer;
            }

            return TensorTree.unflatten(CallReturn, &self.output_tensors);
        }

        /// Release the loaded program.
        pub fn deinit(self: *Self) void {
            self.loaded_program.deinit();
        }
    };
}

/// Derive the return type of a traced function from its comptime signature.
fn DeriveReturnType(comptime func: anytype) type {
    const FnInfo = @typeInfo(@TypeOf(func)).@"fn";
    const RawReturn = FnInfo.return_type.?;

    return switch (@typeInfo(RawReturn)) {
        .error_union => |eu| eu.payload,
        else => RawReturn,
    };
}

test "Compiled.call waits before releasing an execution event" {
    const FakeExecution = struct {
        interface: Executor = .{
            .device = .{ .platform = .cpu },
            .vtable = &vtable,
        },
        waited: bool = false,
        released_event: bool = false,

        const vtable: Executor.VTable = .{
            .upload = upload,
            .download = download,
            .invoke = invoke,
            .await_event = await_event,
            .release_buffer = release_buffer,
            .release_event = release_event,
            .release_program = release_program,
        };

        fn promote(interface: *Executor) *@This() {
            return @fieldParentPtr("interface", interface);
        }

        fn upload(_: *Executor, _: []const u8, _: @import("dtype").DType, _: []const i64) Executor.Error!Executor.Buffer {
            return error.Unsupported;
        }

        fn download(_: *Executor, _: Executor.Buffer, _: []u8) Executor.Error!?Executor.Event {
            return error.Unsupported;
        }

        fn invoke(
            _: *Executor,
            _: *anyopaque,
            _: []const Executor.Buffer,
            outputs: []Executor.Buffer,
            _: Executor.InvokeOptions,
        ) Executor.Error!?Executor.Event {
            outputs[0] = .{ .handle = @ptrFromInt(2) };
            return .{ .handle = @ptrFromInt(3) };
        }

        fn await_event(interface: *Executor, _: Executor.Event) Executor.Error!void {
            promote(interface).waited = true;
        }

        fn release_buffer(_: *Executor, _: Executor.Buffer) void {}

        fn release_event(interface: *Executor, _: Executor.Event) void {
            const self = promote(interface);
            std.debug.assert(self.waited);
            self.released_event = true;
        }

        fn release_program(_: *Executor, _: *anyopaque) void {}
    };
    const Function = struct {
        fn identity(value: Tensor) !Tensor {
            return value;
        }
    };

    const specs = .{Tensor.abstract(.f32, &.{1})};
    var program = pr.Program.init(std.testing.allocator);
    const function = function: {
        var builder = try pr.FunctionBuilder.init(&program, "main");
        defer builder.deinit();
        const input = try builder.param_tensor(.f32, &.{1});
        break :function try program.add_function(try builder.finish(.{ .returns = &.{input} }));
    };
    try program.set_entry(function);
    var traced = Traced(Function.identity, @TypeOf(specs)){
        .program = program,
        .function = function,
    };
    defer traced.deinit();

    var execution: FakeExecution = .{};
    var compiled = try traced.bind(.{
        .executor = &execution.interface,
        .handle = @ptrFromInt(1),
    });
    defer compiled.deinit();

    var inputs = .{Tensor.from_buffer(
        &execution.interface,
        .{ .handle = @ptrFromInt(4) },
        .f32,
        &.{1},
    )};
    defer inputs[0].deinit();
    var result = try compiled.call(&inputs);
    defer result.deinit();

    try std.testing.expect(execution.waited);
    try std.testing.expect(execution.released_event);
}
