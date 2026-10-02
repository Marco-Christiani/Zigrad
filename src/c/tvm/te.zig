//! Tensor-expression bindings.

const std = @import("std");
const ffi = @import("ffi.zig");
const container = @import("container.zig");
const tir = @import("tir.zig");
const DType = @import("../../dtype.zig").DType;

/// Tensor produced by TVM's tensor-expression API.
pub const Tensor = struct {
    object: ffi.Object,

    /// Take a packed result as a tensor expression.
    pub fn take(value: *ffi.OwnedValue) !Tensor {
        try value.borrow().require_instance("te.Tensor");
        return .{ .object = try .take(value) };
    }

    fn as_value(self: *const Tensor) ffi.Value {
        return self.object.as_value();
    }

    /// Release the TVM tensor-expression object.
    pub fn deinit(self: *Tensor) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Matrix multiplication options accepted by TOPI.
pub const MatmulOptions = struct {
    /// Read the left-hand input as transposed.
    transpose_lhs: bool = false,
    /// Read the right-hand input as transposed.
    transpose_rhs: bool = false,
};

/// Create a statically shaped tensor-expression input.
pub fn placeholder(
    /// Allocator used by TVM packed calls.
    allocator: std.mem.Allocator,
    /// Positive dimensions of the tensor.
    shape: []const i64,
    /// Tensor element type.
    dtype: DType,
    /// Name used in generated TIR.
    name: [:0]const u8,
) !Tensor {
    if (shape.len == 0) return error.InvalidTensorShape;

    const dimensions = try allocator.alloc(ffi.Value, shape.len);
    defer allocator.free(dimensions);
    for (shape, dimensions) |dimension, *value| {
        if (dimension <= 0) return error.InvalidTensorShape;
        value.* = .int(dimension);
    }

    var shape_value = try container.Array.from_values(allocator, dimensions);
    defer shape_value.deinit();
    return try ffi.call_global_take(Tensor, allocator, "te.Placeholder", &.{
        shape_value.as_value(),
        ffi.Value.str(dtype_name(dtype)),
        ffi.Value.str(name),
    });
}

/// Create a matrix multiplication tensor expression.
pub fn matmul(
    /// Allocator used by TVM packed calls.
    allocator: std.mem.Allocator,
    /// Left-hand matrix.
    lhs: *const Tensor,
    /// Right-hand matrix.
    rhs: *const Tensor,
    /// Input interpretation options.
    options: MatmulOptions,
) !Tensor {
    return try ffi.call_global_take(Tensor, allocator, "topi.matmul", &.{
        lhs.as_value(),
        rhs.as_value(),
        ffi.Value.boolean(options.transpose_lhs),
        ffi.Value.boolean(options.transpose_rhs),
    });
}

/// Create a TIR function whose parameters and results are `tensors`.
pub fn create_prim_func(
    /// Allocator used by TVM packed calls.
    allocator: std.mem.Allocator,
    /// Tensor expressions in function-parameter order.
    tensors: []const *const Tensor,
) !tir.PrimFunc {
    if (tensors.len == 0) return error.InvalidTensorList;

    const values = try allocator.alloc(ffi.Value, tensors.len);
    defer allocator.free(values);
    for (tensors, values) |tensor, *value| value.* = tensor.as_value();

    var tensor_values = try container.Array.from_values(allocator, values);
    defer tensor_values.deinit();
    return try ffi.call_global_take(
        tir.PrimFunc,
        allocator,
        "te.CreatePrimFunc",
        &.{ tensor_values.as_value(), ffi.Value.none() },
    );
}

fn dtype_name(dtype: DType) [:0]const u8 {
    return switch (dtype) {
        .f16 => "float16",
        .bf16 => "bfloat16",
        .f32 => "float32",
        .f64 => "float64",
        .i8 => "int8",
        .u8 => "uint8",
        .i32 => "int32",
        .i64 => "int64",
        .u32 => "uint32",
        .u64 => "uint64",
        .bool => "bool",
    };
}
