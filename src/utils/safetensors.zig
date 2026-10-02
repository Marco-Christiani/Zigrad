//! Serialize and load tensor containers in the SafeTensors format.
const std = @import("std");
const stz = @import("safetensors_zg");
const Tensor = @import("../tensor.zig");
const DType = @import("pr").DType;

/// Parsed SafeTensors storage accepted by `from_safetensors`.
pub const SafeTensorsFile = stz.SafeTensorsFile;

pub const Opts = struct {
    /// Allocator used for tensor metadata and converted storage.
    allocator: std.mem.Allocator,
    /// Element type of every loaded tensor.
    dtype: DType,
};

/// Serialize a host-backed tensor tree using its field paths as tensor names.
///
///
/// Caller owns the returned SafeTensors bytes.
///
/// TODO: parity with from_safetensors?
pub fn to_safetensors(
    /// A type containing `Tensor` leaves, structs, and/or arrays.
    comptime T: type,
    // Every tensor value must use host storage.
    value: T,
    allocator: std.mem.Allocator,
) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var tensors = std.ArrayList(stz.Tensor).empty;

    try collect(T, value, arena.allocator(), &tensors, "");
    return try stz.serialize_tensors(tensors, allocator);
}

fn collect(
    comptime T: type,
    value: T,
    allocator: std.mem.Allocator,
    tensors: *std.ArrayList(stz.Tensor),
    comptime prefix: []const u8,
) !void {
    if (T == Tensor) {
        const dims = value.shape.const_slice();
        const shape = try allocator.alloc(usize, dims.len);
        for (dims, shape) |dim, *out| out.* = std.math.cast(usize, dim) orelse
            return error.InvalidTensorShape;
        const host_data = value.host_data();
        const data = try allocator.alignedAlloc(u8, .@"8", host_data.len);
        @memcpy(data, host_data);
        try tensors.append(allocator, .{
            .name = if (prefix.len == 0) "tensor" else prefix,
            .dtype = try stz_dtype(value.dtype),
            .shape = shape,
            .data = data,
        });
        return;
    }

    switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.fields) |field| {
            const separator = if (prefix.len == 0) "" else ".";
            try collect(
                field.type,
                @field(value, field.name),
                allocator,
                tensors,
                prefix ++ separator ++ field.name,
            );
        },
        .array => |array| inline for (0..array.len) |index| {
            const separator = if (prefix.len == 0) "" else ".";
            try collect(
                array.child,
                value[index],
                allocator,
                tensors,
                prefix ++ separator ++ std.fmt.comptimePrint("{d}", .{index}),
            );
        },
        else => @compileError("to_safetensors: unsupported type `" ++ @typeName(T) ++
            "` (expected Tensor, struct, or array of same)"),
    }
}

fn stz_dtype(dtype: DType) !stz.Dtype {
    return switch (dtype) {
        .f32 => .f32,
        .f64 => .f64,
        .f16 => .f16,
        .bf16 => .bf16,
        .bool => .bool,
        .i8 => .i8,
        .u8 => .u8,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
    };
}

test stz_dtype {
    inline for (std.meta.fields(DType)) |field| {
        const dtype: DType = @enumFromInt(field.value);
        try std.testing.expectEqualStrings(dtype.name(), @tagName(try stz_dtype(dtype)));
    }
}

/// Load a struct of `Tensor` leaves from a safetensors file.
///
/// Caller provides a struct of type `T` whose field paths mirror
///  the safetensors checkpoint key hierarchy. This function walks `T`
///  at comptime, producing a value of type `T` with every `Tensor`
///  leaf loaded from the `st` file.
///
/// ## Supported field shapes
///
/// - `Tensor`         : required leaf. Looked up at the accumulated field path.
///                       `stz.Error.TensorNotFound` if missing.
/// - `struct { ... }` : recurse into fields, extending the path with the field
///                       names ("." delimeter).
/// - `[N]T`           : recurse into elements, extending the path with
///                      numeric index ("." delimeter).
/// - `?T`             : optional subtree. `stz.Error.TensorNotFound` *anywhere
///                       inside the subtree becomes `null` at the optional
///                       level.* Useful for things such as tied weights:
///                       declare `lm_head: ?struct { weight: Tensor }`
///                       and the consumer resolves the alias at use time.
///
/// Any other type is a compile error.
///
/// NOTE: No extension points right now, if we need different loading behavior per field,
///  consider extending this walker rather than forking at the call site.
///
/// ## Zero-copy semantics
///
/// When the checkpoint's dtype matches the target dtype, each `Tensor` is
///  constructed as a `Tensor.HostSrc.borrow` view into the mmap'd checkpoint bytes.
///
/// When dtypes differ, the walker allocates and converts element-wise
///  (f32/bf16 only). The caller keeps the mmap alive until after the
///  loaded tensors have been uploaded to the device.
pub fn from_safetensors(
    comptime T: type,
    st: *stz.SafeTensorsFile,
    opts: Opts,
) !T {
    return try walk(T, st, opts, "");
}

fn walk(
    comptime T: type,
    st: *stz.SafeTensorsFile,
    opts: Opts,
    comptime prefix: []const u8,
) anyerror!T {
    if (T == Tensor) {
        const view = try st.get(prefix);
        return try tensor_from_view(opts.allocator, opts.dtype, view);
    }
    switch (@typeInfo(T)) {
        .optional => |opt| {
            return walk(opt.child, st, opts, prefix) catch |err| switch (err) {
                stz.Error.TensorNotFound => return null,
                else => return err,
            };
        },
        .@"struct" => |info| {
            var result: T = undefined;
            var initialized: usize = 0;
            errdefer inline for (info.fields, 0..) |field, index| {
                if (index < initialized) deinit_loaded(field.type, &@field(result, field.name));
            };
            inline for (info.fields) |field| {
                const sep = if (prefix.len == 0) "" else ".";
                @field(result, field.name) = try walk(
                    field.type,
                    st,
                    opts,
                    prefix ++ sep ++ field.name,
                );
                initialized += 1;
            }
            return result;
        },
        .array => |arr| {
            var result: T = undefined;
            var initialized: usize = 0;
            errdefer inline for (0..arr.len) |index| {
                if (index < initialized) deinit_loaded(arr.child, &result[index]);
            };
            inline for (0..arr.len) |i| {
                const sep = if (prefix.len == 0) "" else ".";
                result[i] = try walk(
                    arr.child,
                    st,
                    opts,
                    prefix ++ sep ++ std.fmt.comptimePrint("{d}", .{i}),
                );
                initialized += 1;
            }
            return result;
        },
        else => @compileError("from_safetensors: unsupported type `" ++ @typeName(T) ++
            "` (expected Tensor, struct, array, or optional of same)"),
    }
}

fn deinit_loaded(comptime T: type, value: *T) void {
    if (T == Tensor) {
        value.deinit();
        return;
    }
    switch (@typeInfo(T)) {
        .optional => |optional| if (value.*) |*child| deinit_loaded(optional.child, child),
        .@"struct" => |info| inline for (info.fields) |field| {
            deinit_loaded(field.type, &@field(value.*, field.name));
        },
        .array => |array| inline for (0..array.len) |index| {
            deinit_loaded(array.child, &value.*[index]);
        },
        else => unreachable,
    }
}

/// Materialize a `Tensor` from a safetensors `TensorView`.
///
/// Zero-copy borrow when dtypes match, else element-wise allocate-and-convert.
///
/// Cross-dtype conversion currently only `{f32, bf16} <-> {f32, bf16}`.
///  Any other source or target dtype on the mismatch path returns
///  `error.UnsupportedDtypeConversion`.
///
/// The matched-dtype (borrow) path supports the full set in `stz_dtype_matches`.
fn tensor_from_view(
    allocator: std.mem.Allocator,
    target_dtype: DType,
    view: stz.TensorView,
) !Tensor {
    var shape_buf: [8]i64 = undefined;
    if (view.info.shape.len > shape_buf.len) return error.TensorRankUnsupported;
    for (view.info.shape, 0..) |d, i| shape_buf[i] = @intCast(d);
    const shape = shape_buf[0..view.info.shape.len];

    if (stz_dtype_matches(view.info.dtype, target_dtype)) {
        return try Tensor.host(target_dtype, shape, .{ .borrow = view.data });
    }

    if (!convert_supported(view.info.dtype, target_dtype)) {
        return error.UnsupportedDtypeConversion;
    }

    const dst = try Tensor.host(target_dtype, shape, .{ .alloc = allocator });
    const count = dst.shape.num_elements();
    for (0..count) |i| write_element(dst, i, read_element(view, i));
    return dst;
}

/// True iff `tensor_from_view` can convert `src` elements to `dst` dtype
///  element-wise (via the f32 pivot in `read_element`/`write_element`).
inline fn convert_supported(src: stz.Dtype, dst: DType) bool {
    const src_ok = src == .f32 or src == .bf16;
    const dst_ok = dst == .f32 or dst == .bf16;
    return src_ok and dst_ok;
}

inline fn stz_dtype_matches(stz_dt: stz.Dtype, zg_dt: DType) bool {
    return switch (zg_dt) {
        .f32 => stz_dt == .f32,
        .bf16 => stz_dt == .bf16,
        .f16 => stz_dt == .f16,
        .f64 => stz_dt == .f64,
        .i32 => stz_dt == .i32,
        .i64 => stz_dt == .i64,
        inline else => |x| @panic("from_safetensors: unsupported target dtype " ++ @tagName(x)),
    };
}

inline fn read_element(view: stz.TensorView, idx: usize) f32 {
    return switch (view.info.dtype) {
        .f32 => std.mem.bytesAsSlice(f32, view.data)[idx],
        .bf16 => DType.bf16.decode(f32, std.mem.bytesAsSlice(u16, view.data)[idx]),
        else => unreachable,
    };
}

test "structured SafeTensors round trip" {
    const testing = std.testing;
    const Values = struct {
        weight: Tensor,
        nested: struct { bias: Tensor },
    };

    var values = Values{
        .weight = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = testing.allocator }),
        .nested = .{
            .bias = try Tensor.host(.f32, &.{2}, .{ .alloc = testing.allocator }),
        },
    };
    defer values.weight.deinit();
    defer values.nested.bias.deinit();
    @memcpy(values.weight.as_slice(f32), &[_]f32{ 1, 2, 3, 4 });
    @memcpy(values.nested.bias.as_slice(f32), &[_]f32{ 5, 6 });

    const bytes = try to_safetensors(Values, values, testing.allocator);
    defer testing.allocator.free(bytes);
    var file = try stz.SafeTensorsFile.deserialize(bytes, testing.allocator);
    defer file.deinit();
    var restored = try from_safetensors(Values, &file, .{
        .allocator = testing.allocator,
        .dtype = .f32,
    });
    defer restored.weight.deinit();
    defer restored.nested.bias.deinit();

    try testing.expectEqualSlices(f32, values.weight.as_const_slice(f32), restored.weight.as_const_slice(f32));
    try testing.expectEqualSlices(f32, values.nested.bias.as_const_slice(f32), restored.nested.bias.as_const_slice(f32));
}

test "serialize borrowed tensor without eight-byte alignment" {
    const testing = std.testing;
    var storage: [4]u8 align(8) = .{ 0, 1, 2, 3 };
    var tensor = try Tensor.host(.u8, &.{3}, .{ .borrow = storage[1..] });
    defer tensor.deinit();

    const bytes = try to_safetensors(Tensor, tensor, testing.allocator);
    defer testing.allocator.free(bytes);
    var file = try stz.SafeTensorsFile.deserialize(bytes, testing.allocator);
    defer file.deinit();
    const view = try file.get("tensor");
    try testing.expectEqualSlices(u8, storage[1..], view.data);
}

inline fn write_element(buf: Tensor, idx: usize, val: f32) void {
    switch (buf.dtype) {
        .f32 => buf.as_slice(f32)[idx] = val,
        .bf16 => buf.as_slice(u16)[idx] = DType.bf16.encode(f32, val),
        else => unreachable,
    }
}
