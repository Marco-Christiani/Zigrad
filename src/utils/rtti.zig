//! Runtime identity checks for erased pointers.
//!
//! `TypeID` produces a runtime identity for a Zig type within one program image.
//!
//! `TypedPtr` checks reconstruction of an erased pointer. `ErasedBox` adds
//!  a generated cleanup operation for a boxed value.

const std = @import("std");

/// Runtime identity for a Zig type within one program image.
pub const TypeID = enum(usize) {
    _,

    pub fn of(comptime T: type) TypeID {
        return @fromBackingInt(@intCast(@intFromPtr(&Holder(T).id)));
    }

    // TODO(abi): Replace image-local IDs with versioned type identities before
    //  loading passes from DSOs.

    /// Per-type storage whose `id` address serves as the TypeID value.
    ///
    /// The `held = T` decl is load-bearing: it forces the struct body to
    ///  depend on `T`, preventing comptime memoization from returning the
    ///  same struct type (and therefore the same `id` address) for
    ///  different `T`.
    fn Holder(comptime T: type) type {
        return struct {
            const held = T;
            var id: u8 = 0;
        };
    }
};

/// Type-tagged erased pointer.
///
/// Non-owning: ownership and cleanup are the call site's responsibility.
pub const TypedPtr = struct {
    type_id: TypeID,
    raw: *anyopaque,

    /// Wrap a concrete pointer, recording its type identity.
    pub fn init(ptr: anytype) TypedPtr {
        const T = std.meta.Child(@TypeOf(ptr));
        return .{
            .type_id = TypeID.of(T),
            .raw = @ptrCast(ptr),
        };
    }

    /// Reconstruct the concrete pointer.
    ///
    /// Debug and safe builds assert the recorded type identity.
    pub fn cast(self: TypedPtr, comptime T: type) *T {
        std.debug.assert(self.type_id == TypeID.of(T));
        return @ptrCast(@alignCast(self.raw));
    }
};

/// Type-erased boxed value with a generated cleanup operation.
pub const ErasedBox = struct {
    value: TypedPtr,
    allocator: std.mem.Allocator,
    cleanup: *const fn (*ErasedBox) void,

    /// Consume and box `value`.
    ///
    /// Cleanup calls a conventional `T.deinit` when present, then frees the box.
    pub fn init(
        allocator: std.mem.Allocator,
        value: anytype,
    ) std.mem.Allocator.Error!ErasedBox {
        const T = @TypeOf(value);
        var consumed = value;
        errdefer deinit_value(T, &consumed, allocator);

        const boxed = try allocator.create(T);
        boxed.* = consumed;
        return .{
            .value = .init(boxed),
            .allocator = allocator,
            .cleanup = struct {
                fn call(self: *ErasedBox) void {
                    const pointer = self.value.cast(T);
                    deinit_value(T, pointer, self.allocator);
                    self.allocator.destroy(pointer);
                }
            }.call,
        };
    }

    /// Release the contained value and its box.
    pub fn deinit(self: *ErasedBox) void {
        self.cleanup(self);
        self.* = undefined;
    }

    /// Return whether the contained value has type `T`.
    pub fn is(self: ErasedBox, comptime T: type) bool {
        return self.value.type_id == TypeID.of(T);
    }

    /// Borrow the contained value after checking its concrete type.
    pub fn cast(self: *ErasedBox, comptime T: type) *T {
        return self.value.cast(T);
    }

    /// Transfer the contained value and free its box.
    pub fn take(self: *ErasedBox, comptime T: type) T {
        const pointer = self.cast(T);
        const result = pointer.*;
        self.allocator.destroy(pointer);
        self.* = undefined;
        return result;
    }

    fn deinit_value(
        comptime T: type,
        value: *T,
        allocator: std.mem.Allocator,
    ) void {
        if (!comptime has_deinit(T)) return;

        const Deinit = @TypeOf(T.deinit);
        if (comptime Deinit == fn (*T) void) {
            value.deinit();
            return;
        }
        if (comptime Deinit == fn (*T, std.mem.Allocator) void) {
            value.deinit(allocator);
            return;
        }
        @compileError(std.fmt.comptimePrint(
            "{s}.deinit has unsupported type {s}",
            .{ @typeName(T), @typeName(Deinit) },
        ));
    }

    fn has_deinit(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .@"struct", .@"union", .@"enum", .@"opaque" => @hasDecl(T, "deinit"),
            else => false,
        };
    }
};

// Tests.

test TypeID {
    const A = struct { x: i32 };
    const B = struct { y: f64 };

    try std.testing.expectEqual(TypeID.of(A), TypeID.of(A));
    try std.testing.expect(TypeID.of(A) != TypeID.of(B));
}

test TypedPtr {
    const Foo = struct { val: i32 };
    var foo = Foo{ .val = 42 };

    const ptr = TypedPtr.init(&foo);

    const recovered = ptr.cast(Foo);
    try std.testing.expectEqual(42, recovered.val);

    recovered.val = 99;
    try std.testing.expectEqual(99, foo.val);
}

test "ErasedBox deinitializes or transfers its value" {
    const Resource = struct {
        calls: *usize,

        pub fn deinit(self: *@This()) void {
            self.calls.* += 1;
        }
    };

    var cleanup_calls: usize = 0;
    var released = try ErasedBox.init(
        std.testing.allocator,
        Resource{ .calls = &cleanup_calls },
    );
    released.deinit();
    try std.testing.expectEqual(1, cleanup_calls);

    var transferred = try ErasedBox.init(
        std.testing.allocator,
        Resource{ .calls = &cleanup_calls },
    );
    var resource = transferred.take(Resource);
    try std.testing.expectEqual(1, cleanup_calls);
    resource.deinit();
    try std.testing.expectEqual(2, cleanup_calls);

    const Allocated = struct {
        bytes: []u8,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.bytes);
        }
    };
    const bytes = try std.testing.allocator.alloc(u8, 8);
    errdefer std.testing.allocator.free(bytes);
    var allocated = try ErasedBox.init(
        std.testing.allocator,
        Allocated{ .bytes = bytes },
    );
    allocated.deinit();
}
