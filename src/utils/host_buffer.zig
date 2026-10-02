//! Host-side buffer with shape and dtype metadata.
//!
//! Manages CPU-resident data with allocated, borrowed, or memory-mapped backing.
//! `Tensor.to_device` uploads this storage through an `Executor`.
//! TODO(organization): Move this framework value out of `utils`.
const std = @import("std");
const pr = @import("pr");

const DType = pr.DType;
const Shape = pr.Shape;
const BoundedShape = pr.BoundedShape;

pub const HostBuffer = struct {
    /// Backing memory for the buffer. Tagged by allocation strategy
    /// to ensure correct deallocation.
    pub const Backing = union(enum) {
        /// Standard allocator (`alignedAlloc` / `free`).
        heap: struct {
            data: []align(8) u8,
            allocator: std.mem.Allocator,
        },
        /// Memory-mapped file (`mmap` / `munmap`).
        mmap: []align(std.heap.page_size_min) u8,
        /// Non-owning view into externally-managed memory.
        /// `deinit` is a no-op for the data.
        borrowed: []const u8,
    };

    backing: Backing,
    shape: BoundedShape,
    dtype: DType,

    /// Raw byte slice for passing to backends or byte-level operations.
    pub fn data(self: HostBuffer) []const u8 {
        return switch (self.backing) {
            .heap => |h| h.data,
            .mmap => |s| s,
            .borrowed => |s| s,
        };
    }

    /// Mutable byte slice. Only valid for heap-backed buffers.
    pub fn data_mut(self: HostBuffer) []u8 {
        return switch (self.backing) {
            .heap => |h| h.data,
            .mmap => @panic("cannot mutate mmap-backed HostBuffer"),
            .borrowed => @panic("cannot mutate borrowed HostBuffer"),
        };
    }

    pub fn init(allocator: std.mem.Allocator, shape: BoundedShape, dtype: DType) !HostBuffer {
        const num_bytes = shape.num_elements() * dtype.size_in_bytes();
        const bytes = try allocator.alignedAlloc(u8, .@"8", num_bytes);

        return .{
            .backing = .{ .heap = .{ .data = bytes, .allocator = allocator } },
            .shape = shape,
            .dtype = dtype,
        };
    }

    /// Memory-map a file and wrap it as a HostBuffer.
    ///
    /// The returned buffer's data is a read-only view into the mmap'd region.
    /// On `deinit`, the mapping is released via `munmap`.
    pub fn init_mmap(path: []const u8, shape: BoundedShape, dtype: DType) !HostBuffer {
        const raw = try @import("mmap.zig").mmap_file(path);
        return .{
            .backing = .{ .mmap = raw },
            .shape = shape,
            .dtype = dtype,
        };
    }

    /// Borrow a byte slice without releasing it from `deinit`.
    ///
    /// The caller is responsible for keeping `bytes` alive for the
    /// lifetime of this HostBuffer. `deinit` does not free the data.
    pub fn borrow(bytes: []const u8, shape: BoundedShape, dtype: DType) HostBuffer {
        return .{
            .backing = .{ .borrowed = bytes },
            .shape = shape,
            .dtype = dtype,
        };
    }

    pub fn from_slice(allocator: std.mem.Allocator, src: anytype, shape: BoundedShape, dtype: DType) !HostBuffer {
        const DataType = @TypeOf(src);
        const data_info = @typeInfo(DataType);

        if (data_info != .pointer) {
            @compileError("from_slice requires a pointer or slice type, got: " ++ @typeName(DataType));
        }

        if (data_info.pointer.size != .slice and data_info.pointer.size != .one) {
            @compileError("from_slice requires a slice or pointer-to-array, got pointer size: " ++ @tagName(data_info.pointer.size));
        }

        const num_bytes = shape.num_elements() * dtype.size_in_bytes();
        const data_bytes = try allocator.alignedAlloc(u8, .@"8", num_bytes);

        const src_bytes = std.mem.sliceAsBytes(src);
        if (src_bytes.len < num_bytes) return error.InsufficientData;
        @memcpy(data_bytes, src_bytes[0..num_bytes]);

        return .{
            .backing = .{ .heap = .{ .data = data_bytes, .allocator = allocator } },
            .shape = shape,
            .dtype = dtype,
        };
    }

    /// Release heap and memory-mapped storage.
    ///
    /// Borrowed storage remains with its caller.
    pub fn deinit(self: *HostBuffer) void {
        switch (self.backing) {
            .heap => |h| h.allocator.free(h.data),
            .mmap => |s| @import("mmap.zig").munmap(s),
            .borrowed => {},
        }
        self.* = undefined;
    }

    /// Fill buffer with a scalar value. Heap-backed only.
    pub fn fill(self: *HostBuffer, comptime T: type, value: T) void {
        const count = self.shape.num_elements();
        const slice: []T = @alignCast(std.mem.bytesAsSlice(T, self.data_mut()));
        for (slice[0..count]) |*elem| {
            elem.* = value;
        }
    }

    /// View buffer as mutable elements of `T`. Heap-backed only.
    pub fn as_slice(self: *HostBuffer, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.data_mut()));
    }

    /// Print buffer contents (for debugging).
    pub fn print(self: *HostBuffer, writer: anytype, allocator: std.mem.Allocator) !void {
        const shape_str = try self.shape.format(allocator);
        defer allocator.free(shape_str);

        try writer.print("HostBuffer({s}, {s}): ", .{ self.dtype.name(), shape_str });

        const max_print = @min(self.shape.num_elements(), 16);

        switch (self.dtype) {
            .bf16 => {
                const slice: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.data()));
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.2}", .{DType.bf16.decode(f32, val)});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            .f32 => {
                const slice: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, self.data()));
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.2}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            .f64 => {
                const slice: []const f64 = @alignCast(std.mem.bytesAsSlice(f64, self.data()));
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d:.2}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            .i32 => {
                const slice: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, self.data()));
                try writer.writeAll("[");
                for (slice[0..max_print], 0..) |val, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{val});
                }
                if (max_print < self.shape.num_elements()) {
                    try writer.writeAll(", ...");
                }
                try writer.writeAll("]");
            },
            else => try writer.writeAll("<unsupported dtype for print>"),
        }

        try writer.writeAll("\n");
    }
};

test "HostBuffer basic operations" {
    const allocator = std.testing.allocator;

    var buf = try HostBuffer.init(allocator, BoundedShape.from_slice(&[_]i64{ 2, 3 }), .f32);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 6), buf.shape.num_elements());
    try std.testing.expectEqual(@as(usize, 24), buf.data().len);

    buf.fill(f32, 42.0);
    const slice = buf.as_slice(f32);
    try std.testing.expectEqual(@as(f32, 42.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 42.0), slice[5]);
}

test "HostBuffer from_slice" {
    const allocator = std.testing.allocator;

    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf = try HostBuffer.from_slice(allocator, &src, BoundedShape.from_slice(&[_]i64{4}), .f32);
    defer buf.deinit();

    const slice = buf.as_slice(f32);
    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice[3]);
}

test "HostBuffer borrow" {
    const src = [_]f32{ 1.0, 2.0, 3.0 };
    var buf = HostBuffer.borrow(std.mem.sliceAsBytes(&src), BoundedShape.from_slice(&[_]i64{3}), .f32);
    defer buf.deinit(); // no-op for borrowed

    try std.testing.expectEqual(@as(usize, 12), buf.data().len);
}
