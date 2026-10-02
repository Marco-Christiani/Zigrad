//! Runtime composition of compiler passes over open Zig types.

const std = @import("std");
const Context = @import("context.zig").Context;
const operation = @import("operation.zig");
const rtti = @import("rtti");
const ErasedBox = rtti.ErasedBox;

const log = std.log.scoped(.@"zg/pipeline");

/// A mutable FIFO queue of compiler passes.
///
/// Pass implementations are copied into queue-owned boxes. Pass a pointer when
///  mutable state or identity must remain caller-managed. Each successful run
///  consumes the prefix that produces its requested output type.
pub const Pipeline = struct {
    const Entry = struct {
        input_type_id: rtti.TypeID,
        input_type_name: []const u8,
        output_type_id: rtti.TypeID,
        output_type_name: []const u8,
        implementation: ErasedBox,
        invoke: *const fn (*Entry, *ErasedBox, *Context) anyerror!void,
    };

    /// Failures while adding a pass to a pipeline.
    pub const AddError = std.mem.Allocator.Error || error{IncompatiblePass};

    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    head: usize = 0,

    /// Initialize an empty pass queue.
    pub fn init(allocator: std.mem.Allocator) Pipeline {
        return .{ .allocator = allocator };
    }

    /// Release pass storage remaining in this queue.
    pub fn deinit(self: *Pipeline) void {
        for (self.entries.items[self.head..]) |*entry| entry.implementation.deinit();
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Append an operation after checking its declared type edge.
    ///
    /// On success, a type-changing operation must not retain its input. On
    ///  failure, every operation must leave its input valid for release. A
    ///  same-type operation transfers the current value's resources to its output.
    pub fn add(
        self: *Pipeline,
        implementation: anytype,
    ) AddError!void {
        const Stored = @TypeOf(implementation);
        const Operation = operation.type_of(Stored);
        comptime operation.validate(Operation);
        const Input = Operation.Input;
        const Output = Operation.Output;

        if (self.entries.items.len > 0) {
            const preceding = self.entries.items[self.entries.items.len - 1];
            if (preceding.output_type_id != rtti.TypeID.of(Input)) {
                log.debug(
                    "cannot append {s} -> {s} after a pass producing {s}",
                    .{ @typeName(Input), @typeName(Output), preceding.output_type_name },
                );
                return error.IncompatiblePass;
            }
        }

        var stored = try ErasedBox.init(self.allocator, implementation);
        errdefer stored.deinit();

        const Adapter = struct {
            fn invoke(entry: *Entry, current: *ErasedBox, context: *Context) anyerror!void {
                if (!current.is(Input)) return error.TypeMismatch;

                const input = current.cast(Input);
                const concrete = entry.implementation.cast(Stored);
                const output = if (comptime @typeInfo(Stored) == .pointer)
                    try concrete.*.run(input.*, context)
                else
                    try concrete.run(input.*, context);

                if (comptime Input == Output) {
                    input.* = output;
                    return;
                }

                const next = try ErasedBox.init(context.allocator, output);
                current.deinit();
                current.* = next;
            }
        };

        try self.entries.append(self.allocator, .{
            .input_type_id = rtti.TypeID.of(Input),
            .input_type_name = @typeName(Input),
            .output_type_id = rtti.TypeID.of(Output),
            .output_type_name = @typeName(Output),
            .implementation = stored,
            .invoke = Adapter.invoke,
        });
    }

    /// Run through the requested output phase and consume that queue prefix.
    ///
    /// The pipeline consumes `input` after its type matches the first pass. Pass
    ///  a pointer when the referenced value must remain caller-managed. When a
    ///  pass fails after execution begins, only `deinit` is valid on the pipeline.
    pub fn run(
        self: *Pipeline,
        comptime Output: type,
        input: anytype,
        context: *Context,
    ) anyerror!Output {
        const Input = @TypeOf(input);
        if (self.head < self.entries.items.len and
            rtti.TypeID.of(Input) != self.entries.items[self.head].input_type_id)
        {
            log.debug(
                "pipeline expects {s}, received {s}",
                .{ self.entries.items[self.head].input_type_name, @typeName(Input) },
            );
            return error.InputTypeMismatch;
        }
        const stop = try self.stop_index(Input, Output);

        var current = try ErasedBox.init(context.allocator, input);
        errdefer current.deinit();
        for (self.entries.items[self.head..stop]) |*entry| {
            try entry.invoke(entry, &current, context);
        }
        const output = current.take(Output);

        for (self.entries.items[self.head..stop]) |*entry| {
            entry.implementation.deinit();
        }
        self.head = stop;
        return output;
    }

    fn stop_index(
        self: *const Pipeline,
        comptime Input: type,
        comptime Output: type,
    ) error{OutputTypeMismatch}!usize {
        if (self.head == self.entries.items.len) {
            if (comptime Input == Output) return self.head;
            return error.OutputTypeMismatch;
        }

        if (comptime Input == Output) {
            if (self.entries.items[self.head].output_type_id != rtti.TypeID.of(Output)) {
                return self.head;
            }
        }

        var stop = self.head;
        while (stop < self.entries.items.len) : (stop += 1) {
            if (self.entries.items[stop].output_type_id != rtti.TypeID.of(Output)) continue;

            stop += 1;
            while (stop < self.entries.items.len and
                self.entries.items[stop].output_type_id == rtti.TypeID.of(Output))
            {
                stop += 1;
            }
            return stop;
        }
        return error.OutputTypeMismatch;
    }
};

test "Pipeline runs an open heterogeneous pass queue" {
    const Source = struct { value: u32 };
    const Intermediate = struct { value: u64 };
    const Artifact = struct { value: u64 };

    const Increment = struct {
        pub const Input = Source;
        pub const Output = Source;

        amount: u32,

        pub fn run(self: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value + self.amount };
        }
    };
    const Lower = struct {
        pub const Input = Source;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value };
        }
    };
    const Compile = struct {
        pub const Input = Intermediate;
        pub const Output = Artifact;

        factor: u64,

        pub fn run(self: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value * self.factor };
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Increment{ .amount = 3 });
    try pipeline.add(Lower{});
    try pipeline.add(Compile{ .factor = 4 });

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    const artifact = try pipeline.run(
        Artifact,
        Source{ .value = 2 },
        &context,
    );

    try std.testing.expectEqual(@as(u64, 20), artifact.value);
}

test "Pipeline consumes one requested output phase at a time" {
    const Source = struct { value: u32 };
    const Intermediate = struct { value: u64 };
    const Artifact = struct { value: u64 };

    const Increment = struct {
        pub const Input = Source;
        pub const Output = Source;

        amount: u32,

        pub fn run(self: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value + self.amount };
        }
    };
    const Lower = struct {
        pub const Input = Source;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value };
        }
    };
    const Package = struct {
        pub const Input = Artifact;
        pub const Output = u128;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input.value;
        }
    };
    const Double = struct {
        pub const Input = Intermediate;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value * 2 };
        }
    };
    const Compile = struct {
        pub const Input = Intermediate;
        pub const Output = Artifact;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return .{ .value = input.value };
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Increment{ .amount = 1 });
    try pipeline.add(Increment{ .amount = 2 });
    try pipeline.add(Lower{});
    try pipeline.add(Double{});
    try pipeline.add(Compile{});

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    const source = try pipeline.run(Source, Source{ .value = 4 }, &context);
    try std.testing.expectEqual(@as(u32, 7), source.value);
    try std.testing.expectEqual(@as(usize, 2), pipeline.head);

    const intermediate = try pipeline.run(Intermediate, source, &context);
    try std.testing.expectEqual(@as(u64, 14), intermediate.value);
    try std.testing.expectEqual(@as(usize, 4), pipeline.head);

    const artifact = try pipeline.run(Artifact, intermediate, &context);
    try std.testing.expectEqual(@as(u64, 14), artifact.value);
    try std.testing.expectEqual(pipeline.entries.items.len, pipeline.head);

    try pipeline.add(Package{});
    const packaged = try pipeline.run(u128, artifact, &context);
    try std.testing.expectEqual(@as(u128, 14), packaged);
}

test "Pipeline rejects incompatible adjacent types" {
    const First = struct {
        pub const Input = u32;
        pub const Output = u64;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };
    const Wrong = struct {
        pub const Input = u16;
        pub const Output = u16;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(First{});
    try std.testing.expectError(error.IncompatiblePass, pipeline.add(Wrong{}));
}

test "Pipeline checks run endpoint types before execution" {
    const Convert = struct {
        pub const Input = u32;
        pub const Output = u64;

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }
    };

    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Convert{});

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    try std.testing.expectError(
        error.InputTypeMismatch,
        pipeline.run(u64, @as(u16, 1), &context),
    );
    try std.testing.expectError(
        error.OutputTypeMismatch,
        pipeline.run(u16, @as(u32, 1), &context),
    );
}

test "Pipeline releases stored pass values" {
    const Pass = struct {
        pub const Input = u32;
        pub const Output = u32;

        calls: *usize,

        pub fn run(_: *@This(), input: Input, _: *Context) !Output {
            return input;
        }

        pub fn deinit(self: *@This()) void {
            self.calls.* += 1;
        }
    };

    var calls: usize = 0;
    var pipeline = Pipeline.init(std.testing.allocator);
    try pipeline.add(Pass{ .calls = &calls });
    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    _ = try pipeline.run(u32, @as(u32, 1), &context);
    try std.testing.expectEqual(1, calls);
    pipeline.deinit();
    try std.testing.expectEqual(1, calls);
}

test "Pipeline releases the current intermediate after failure" {
    const Intermediate = struct {
        bytes: []u8,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.bytes);
        }
    };
    const Lower = struct {
        pub const Input = u32;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), input: Input, context: *Context) !Output {
            const bytes = try context.allocator.alloc(u8, input);
            return .{ .bytes = bytes };
        }
    };
    const Reject = struct {
        pub const Input = Intermediate;
        pub const Output = Intermediate;

        pub fn run(_: *@This(), _: Input, _: *Context) !Output {
            return error.Rejected;
        }
    };
    var pipeline = Pipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    try pipeline.add(Lower{});
    try pipeline.add(Reject{});

    var context: Context = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    try std.testing.expectError(
        error.Rejected,
        pipeline.run(Intermediate, @as(u32, 8), &context),
    );
}
