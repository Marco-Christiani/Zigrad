//! Runtime loading and execution contracts for framework values.
//!
//! Compiler inputs, emitted representations, device discovery, artifact
//!  serialization, and integration-specific dispatch configuration remain with
//!  the integration that implements them.

const std = @import("std");
const device_mod = @import("device");
const DType = @import("dtype").DType;

const Executor = @This();

/// Device selected by this executor.
device: device_mod.Device,

/// Dispatch table supplied by the execution integration.
vtable: *const VTable,

/// Failures exposed by the runtime execution contract.
pub const Error = error{
    OutOfMemory,
    InvalidArgument,
    Unsupported,
    ResourceExhausted,
    Unavailable,
    IntegrationFailure,
};

/// Operations required by `Tensor`, callable binding, and training helpers.
///
/// An integration embeds `Executor`, supplies a static vtable, and recovers its
///  concrete state with `@fieldParentPtr`.
pub const VTable = struct {
    upload: *const fn (
        executor: *Executor,
        data: []const u8,
        dtype: DType,
        shape: []const i64,
    ) Error!Buffer,
    download: *const fn (
        executor: *Executor,
        buffer: Buffer,
        destination: []u8,
    ) Error!?Event,
    invoke: *const fn (
        executor: *Executor,
        program_handle: *anyopaque,
        inputs: []const Buffer,
        outputs: []Buffer,
        options: InvokeOptions,
    ) Error!?Event,
    await_event: *const fn (executor: *Executor, event: Event) Error!void,
    release_buffer: *const fn (executor: *Executor, buffer: Buffer) void,
    release_event: *const fn (executor: *Executor, event: Event) void,
    release_program: *const fn (executor: *Executor, program_handle: *anyopaque) void,
};

/// Handle to storage managed by an execution integration.
pub const Buffer = struct {
    /// Opaque storage handle released through `Executor.release`.
    handle: *anyopaque,
};

/// Program loaded by one executor.
pub const LoadedProgram = struct {
    /// Executor that loaded the program and accepts its handles.
    executor: *Executor,

    /// Opaque program handle released by `deinit`.
    handle: *anyopaque,

    /// Release this program through the executor that loaded it.
    pub fn deinit(self: *LoadedProgram) void {
        self.executor.release_program(self);
    }
};

/// Handle to optional asynchronous completion state.
pub const Event = struct {
    /// Opaque completion handle released through `Executor.release_event`.
    handle: *anyopaque,
};

/// Information that may affect one invocation without changing its results.
pub const InvokeOptions = struct {
    /// Flattened input positions available for buffer reuse.
    donated_input_indices: []const usize = &.{},
};

/// Type-erased loader for one compiler artifact type.
///
/// Loading consumes the artifact on success. The caller retains the artifact
///  when loading fails.
pub fn Loader(comptime ArtifactType: type) type {
    return struct {
        const Self = @This();

        /// Executor associated with programs produced by this loader.
        executor: *Executor,

        /// Dispatch table supplied by the loader implementation.
        vtable: *const Self.VTable,

        /// Compiler artifact consumed by this loader.
        pub const Artifact = ArtifactType;

        /// Function dispatched by the type-erased loader.
        pub const VTable = struct {
            load: *const fn (
                loader: *Self,
                artifact: *Artifact,
            ) Error!*anyopaque,
        };

        /// Consume `artifact` and return a program associated with `executor`.
        pub fn load(
            self: *Self,
            artifact: *Artifact,
        ) Error!LoadedProgram {
            const handle = try self.vtable.load(self, artifact);
            return .{
                .executor = self.executor,
                .handle = handle,
            };
        }
    };
}

/// Copy host data into integration-managed storage.
pub fn upload(
    self: *Executor,
    data: []const u8,
    dtype: DType,
    shape: []const i64,
) Error!Buffer {
    return try self.vtable.upload(self, data, dtype, shape);
}

/// Copy a buffer into caller-provided host memory.
///
/// A null event means the copy completed before this function returned.
pub fn download(
    self: *Executor,
    buffer: Buffer,
    destination: []u8,
) Error!?Event {
    return try self.vtable.download(self, buffer, destination);
}

/// Execute a program and populate every output slot.
///
/// `inputs`, `outputs`, and `program` must come from this executor. A null
///  event means execution completed synchronously or exposes no completion
///  event.
pub fn invoke(
    self: *Executor,
    program: LoadedProgram,
    inputs: []const Buffer,
    outputs: []Buffer,
    options: InvokeOptions,
) Error!?Event {
    if (program.executor != self) return error.InvalidArgument;
    return try self.vtable.invoke(self, program.handle, inputs, outputs, options);
}

/// Wait until an asynchronous operation completes.
pub fn wait(self: *Executor, event: Event) Error!void {
    return try self.vtable.await_event(self, event);
}

/// Release one buffer.
pub fn release(self: *Executor, buffer: Buffer) void {
    self.vtable.release_buffer(self, buffer);
}

/// Release one event.
pub fn release_event(self: *Executor, event: Event) void {
    self.vtable.release_event(self, event);
}

/// Release one loaded program.
pub fn release_program(self: *Executor, program: *LoadedProgram) void {
    std.debug.assert(program.executor == self);
    self.vtable.release_program(self, program.handle);
    program.* = undefined;
}

test "Executor dispatches through an external implementation" {
    const Fake = struct {
        interface: Executor = .{
            .device = .{ .platform = .cpu },
            .vtable = &vtable,
        },
        invoked: bool = false,
        released: bool = false,

        const vtable: VTable = .{
            .upload = upload_impl,
            .download = download_impl,
            .invoke = invoke_impl,
            .await_event = await_impl,
            .release_buffer = release_buffer_impl,
            .release_event = release_event_impl,
            .release_program = release_program_impl,
        };

        fn promote(interface: *Executor) *@This() {
            return @fieldParentPtr("interface", interface);
        }

        fn upload_impl(
            _: *Executor,
            data: []const u8,
            _: DType,
            _: []const i64,
        ) Error!Buffer {
            return .{ .handle = @ptrCast(@constCast(data.ptr)) };
        }

        fn download_impl(
            _: *Executor,
            _: Buffer,
            _: []u8,
        ) Error!?Event {
            return null;
        }

        fn invoke_impl(
            interface: *Executor,
            _: *anyopaque,
            _: []const Buffer,
            outputs: []Buffer,
            _: InvokeOptions,
        ) Error!?Event {
            promote(interface).invoked = true;
            for (outputs) |*output| output.* = .{ .handle = @ptrFromInt(1) };
            return null;
        }

        fn await_impl(_: *Executor, _: Event) Error!void {}
        fn release_buffer_impl(_: *Executor, _: Buffer) void {}
        fn release_event_impl(_: *Executor, _: Event) void {}
        fn release_program_impl(interface: *Executor, _: *anyopaque) void {
            promote(interface).released = true;
        }
    };

    const Artifact = struct { handle: *anyopaque };
    const LoaderInterface = Loader(Artifact);
    const FakeLoader = struct {
        interface: LoaderInterface,

        const vtable: LoaderInterface.VTable = .{
            .load = load,
        };

        fn init(executor: *Executor) @This() {
            return .{
                .interface = .{
                    .executor = executor,
                    .vtable = &vtable,
                },
            };
        }

        fn load(_: *LoaderInterface, artifact: *Artifact) Error!*anyopaque {
            const handle = artifact.handle;
            artifact.* = undefined;
            return handle;
        }
    };

    var fake: Fake = .{};
    var outputs: [1]Buffer = undefined;
    const event = try fake.interface.invoke(
        .{ .executor = &fake.interface, .handle = @ptrFromInt(1) },
        &.{},
        &outputs,
        .{},
    );

    try std.testing.expect(fake.invoked);
    try std.testing.expect(event == null);
    try std.testing.expectEqual(@as(usize, 1), @intFromPtr(outputs[0].handle));

    var loader = FakeLoader.init(&fake.interface);
    var artifact = Artifact{ .handle = @ptrFromInt(2) };
    var program = try loader.interface.load(&artifact);
    try std.testing.expect(program.executor == &fake.interface);
    try std.testing.expectEqual(@as(usize, 2), @intFromPtr(program.handle));
    program.deinit();
    try std.testing.expect(fake.released);
}
