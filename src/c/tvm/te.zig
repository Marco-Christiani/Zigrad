//! Tensor-expression bindings.

const std = @import("std");
const ffi = @import("ffi.zig");
const container = @import("container.zig");
const tir = @import("tir.zig");
const DType = @import("dtype").DType;

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

/// Matrix multiplication arithmetic and input interpretation.
pub const MatmulOptions = struct {
    /// Products and reduction use f32.
    accumulation_dtype: enum { f32 } = .f32,
    /// Conversion applied once after the complete reduction.
    output_dtype: DType = .f32,
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
    const accumulation_dtype: [:0]const u8 = switch (options.accumulation_dtype) {
        .f32 => "float32",
    };
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Packed results own references until the expression graph is complete.
    var expressions: std.ArrayList(ffi.OwnedValue) = .empty;
    defer {
        for (expressions.items) |*expression| expression.deinit();
        expressions.deinit(a);
    }
    const builder = ExpressionBuilder{ .allocator = a, .values = &expressions };
    var lhs_shape = try ffi.get_field(lhs.as_value(), "shape");
    defer lhs_shape.deinit();
    var rhs_shape = try ffi.get_field(rhs.as_value(), "shape");
    defer rhs_shape.deinit();
    var ls = try container.Array.take(&lhs_shape);
    defer ls.deinit();
    var rs = try container.Array.take(&rhs_shape);
    defer rs.deinit();
    if (try ls.len(a) != 2 or try rs.len(a) != 2) return error.InvalidTensorShape;
    var m = try ls.get(a, if (options.transpose_lhs) 1 else 0);
    defer m.deinit();
    var k = try ls.get(a, if (options.transpose_lhs) 0 else 1);
    defer k.deinit();
    var n = try rs.get(a, if (options.transpose_rhs) 0 else 1);
    defer n.deinit();
    const zero = try builder.call("ir.IntImm", &.{ .str("int32"), .int(0), .none() });
    const i = try builder.call("tir.Var", &.{ .str("i"), .str("int32"), .none() });
    const j = try builder.call("tir.Var", &.{ .str("j"), .str("int32"), .none() });
    const r = try builder.call("tir.Var", &.{ .str("k"), .str("int32"), .none() });
    const axis_i = try builder.axis(zero, m.borrow(), i, 0);
    const axis_j = try builder.axis(zero, n.borrow(), j, 0);
    const axis_r = try builder.axis(zero, k.borrow(), r, 2);
    const lhs_indices = if (options.transpose_lhs) try builder.array(&.{ r, i }) else try builder.array(&.{ i, r });
    const rhs_indices = if (options.transpose_rhs) try builder.array(&.{ j, r }) else try builder.array(&.{ r, j });
    const load_a = try builder.call("tir.ProducerLoad", &.{ lhs.as_value(), lhs_indices, .none() });
    const load_b = try builder.call("tir.ProducerLoad", &.{ rhs.as_value(), rhs_indices, .none() });
    const cast_a = try builder.call("tir.Cast", &.{ .str(accumulation_dtype), load_a, .none() });
    const cast_b = try builder.call("tir.Cast", &.{ .str(accumulation_dtype), load_b, .none() });
    const product = try builder.call("tir.Mul", &.{ cast_a, cast_b, .none() });
    const x = try builder.call("tir.Var", &.{ .str("x"), .str(accumulation_dtype), .none() });
    const y = try builder.call("tir.Var", &.{ .str("y"), .str(accumulation_dtype), .none() });
    const sum = try builder.call("tir.Add", &.{ x, y, .none() });
    const identity = try builder.call("ir.FloatImm", &.{ .str(accumulation_dtype), .float(0), .none() });
    const reducer = try builder.call("tir.CommReducer", &.{
        try builder.array(&.{x}),   try builder.array(&.{y}),
        try builder.array(&.{sum}), try builder.array(&.{identity}),
        .none(),
    });
    const reduction = try builder.call("tir.Reduce", &.{
        reducer, try builder.array(&.{product}), try builder.array(&.{axis_r}),
        .none(), .int(0),                        try builder.array(&.{}),
        .none(),
    });
    const axes = try builder.array(&.{ axis_i, axis_j });
    const operation = try builder.call("te.ComputeOp", &.{
        .str("accumulator"), .str("matmul"), .none(), axes, try builder.array(&.{reduction}),
    });
    var accumulated = try ffi.call_global_take(Tensor, a, "te.OpGetOutput", &.{ operation, .int(0) });
    defer accumulated.deinit();
    if (options.output_dtype == .f32) {
        // The result retains its graph after the temporary references are freed.
        return try ffi.call_global_take(Tensor, allocator, "te.OpGetOutput", &.{ operation, .int(0) });
    }
    const load = try builder.call("tir.ProducerLoad", &.{ accumulated.as_value(), try builder.array(&.{ i, j }), .none() });
    const converted = try builder.call("tir.Cast", &.{ .str(dtype_name(options.output_dtype)), load, .none() });
    const output_op = try builder.call("te.ComputeOp", &.{
        .str("output"), .str("injective"), .none(), axes, try builder.array(&.{converted}),
    });
    return try ffi.call_global_take(Tensor, allocator, "te.OpGetOutput", &.{ output_op, .int(0) });
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

const ExpressionBuilder = struct {
    allocator: std.mem.Allocator,
    values: *std.ArrayList(ffi.OwnedValue),

    fn call(self: ExpressionBuilder, name: []const u8, args: []const ffi.Value) !ffi.Value {
        var value = try ffi.call_global(self.allocator, name, args);
        errdefer value.deinit();
        const borrowed = value.borrow();
        try self.values.append(self.allocator, value);
        return borrowed;
    }

    fn array(self: ExpressionBuilder, values: []const ffi.Value) !ffi.Value {
        var array_value = try container.Array.from_values(self.allocator, values);
        defer array_value.deinit();
        var owned = try ffi.OwnedValue.retain(array_value.as_value());
        errdefer owned.deinit();
        const borrowed = owned.borrow();
        try self.values.append(self.allocator, owned);
        return borrowed;
    }

    fn axis(self: ExpressionBuilder, zero: ffi.Value, extent: ffi.Value, variable: ffi.Value, kind: i64) !ffi.Value {
        const range = try self.call("ir.Range_from_min_extent", &.{ zero, extent, .none() });
        return self.call("tir.IterVar", &.{ range, variable, .int(kind), .str(""), .none() });
    }
};
