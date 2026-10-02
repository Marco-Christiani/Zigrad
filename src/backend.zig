//! Terminal compiler, loader, and executor composition.

const std = @import("std");

const compilation = @import("compilation");
const Executor = @import("execution.zig");

/// Failures exposed by terminal compilation, loading, and execution.
pub const Error = compilation.Error || Executor.Error;

/// Type-erased terminal backend accepting one compiler input type.
///
/// Implementations compose a compiler and loader. Source transforms and
///  lowerings remain ordinary operations outside this interface.
pub fn Backend(comptime InputType: type) type {
    return struct {
        const Self = @This();

        /// Compiler input accepted by this backend.
        pub const Input = InputType;

        /// Program loaded into this backend's executor.
        pub const Output = Executor.LoadedProgram;

        /// Executor that runs programs prepared by this backend.
        executor: *Executor,

        /// Dispatch table supplied by the backend implementation.
        vtable: *const VTable,

        /// Function dispatched by the type-erased backend.
        pub const VTable = struct {
            prepare: *const fn (
                backend: *Self,
                input: Input,
                context: *compilation.Context,
            ) Error!Output,
        };

        /// Compile and load one compiler input.
        pub fn run(
            self: *Self,
            input: Input,
            context: *compilation.Context,
        ) Error!Output {
            if (context.device) |selected| {
                if (!selected.eql(self.executor.device)) return error.InvalidArgument;
            }

            var execution_context = context.*;
            execution_context.device = self.executor.device;
            var program = try self.vtable.prepare(self, input, &execution_context);
            if (program.executor != self.executor) {
                program.deinit();
                return error.InvalidArgument;
            }
            return program;
        }
    };
}

test "Backend implementations share one selectable contract" {
    const Input = struct { value: u32 };
    const Interface = Backend(Input);

    const FakeExecution = struct {
        interface: Executor = .{
            .device = .{ .platform = .cpu },
            .vtable = &vtable,
        },

        const vtable: Executor.VTable = .{
            .upload = upload,
            .download = download,
            .invoke = invoke,
            .await_event = wait,
            .release_buffer = release_buffer,
            .release_event = release_event,
            .release_program = release_program,
        };

        fn upload(_: *Executor, _: []const u8, _: @import("dtype").DType, _: []const i64) Executor.Error!Executor.Buffer {
            return error.Unsupported;
        }
        fn download(_: *Executor, _: Executor.Buffer, _: []u8) Executor.Error!?Executor.Event {
            return error.Unsupported;
        }
        fn invoke(_: *Executor, _: *anyopaque, _: []const Executor.Buffer, _: []Executor.Buffer, _: Executor.InvokeOptions) Executor.Error!?Executor.Event {
            return error.Unsupported;
        }
        fn wait(_: *Executor, _: Executor.Event) Executor.Error!void {
            return error.Unsupported;
        }
        fn release_buffer(_: *Executor, _: Executor.Buffer) void {}
        fn release_event(_: *Executor, _: Executor.Event) void {}
        fn release_program(_: *Executor, _: *anyopaque) void {}
    };

    const FakeBackend = struct {
        interface: Interface,
        handle: *anyopaque,

        const vtable: Interface.VTable = .{
            .prepare = prepare,
        };

        fn init(executor: *Executor, handle: *anyopaque) @This() {
            return .{
                .interface = .{
                    .executor = executor,
                    .vtable = &vtable,
                },
                .handle = handle,
            };
        }

        fn prepare(
            interface: *Interface,
            _: Input,
            _: *compilation.Context,
        ) Error!Executor.LoadedProgram {
            const self: *@This() = @fieldParentPtr("interface", interface);
            return .{
                .executor = self.interface.executor,
                .handle = self.handle,
            };
        }
    };

    var execution: FakeExecution = .{};
    var first = FakeBackend.init(&execution.interface, @ptrFromInt(1));
    var second = FakeBackend.init(&execution.interface, @ptrFromInt(2));
    const selected = struct {
        fn choose(
            first_backend: *Interface,
            second_backend: *Interface,
            select_second: bool,
        ) *Interface {
            return if (select_second) second_backend else first_backend;
        }
    }.choose(&first.interface, &second.interface, true);
    var context: compilation.Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    context.device = .{ .platform = .cuda };
    try std.testing.expectError(
        error.InvalidArgument,
        selected.run(.{ .value = 1 }, &context),
    );

    context.device = null;
    var program = try selected.run(.{ .value = 1 }, &context);
    defer program.deinit();

    try std.testing.expect(program.executor == &execution.interface);
    try std.testing.expectEqual(@as(usize, 2), @intFromPtr(program.handle));
}
