// StableHLO Dialect for MLIR
// Adapted from ZML (https://github.com/zml/zml)
// Original Copyright (c) 2024 ZML Contributors
// Apache License 2.0

const std = @import("std");
const mlir = @import("../mlir.zig");

// Use the shared c namespace from mlir.zig to avoid type incompatibility
const c = mlir.c;

fn BoundedArray(comptime T: type, comptime capacity: usize) type {
    return struct {
        buffer: [capacity]T = undefined,
        len: usize = 0,

        const Self = @This();

        pub fn append_assume_capacity(self: *Self, item: T) void {
            self.buffer[self.len] = item;
            self.len += 1;
        }

        pub fn append_slice_assume_capacity(self: *Self, items: []const T) void {
            @memcpy(self.buffer[self.len..][0..items.len], items);
            self.len += items.len;
        }

        pub fn append_slice(self: *Self, items: []const T) error{Overflow}!void {
            if (self.len + items.len > capacity) return error.Overflow;
            self.append_slice_assume_capacity(items);
        }

        pub fn const_slice(self: *const Self) []const T {
            return self.buffer[0..self.len];
        }
    };
}

pub const abs = UnaryFn("stablehlo.abs").call;
pub const cosine = UnaryFn("stablehlo.cosine").call;
pub const sine = UnaryFn("stablehlo.sine").call;
pub const exponential = UnaryFn("stablehlo.exponential").call;
pub const exponential_minus_one = UnaryFn("stablehlo.exponential_minus_one").call;
pub const floor = UnaryFn("stablehlo.floor").call;
pub const log = UnaryFn("stablehlo.log").call;
pub const log_plus_one = UnaryFn("stablehlo.log_plus_one").call;
pub const not = UnaryFn("stablehlo.not").call;
pub const negate = UnaryFn("stablehlo.negate").call;
pub const sqrt = UnaryFn("stablehlo.sqrt").call;
pub const tanh = UnaryFn("stablehlo.tanh").call;
pub const cbrt = UnaryFn("stablehlo.cbrt").call;
pub const ceil = UnaryFn("stablehlo.ceil").call;
pub const rsqrt = UnaryFn("stablehlo.rsqrt").call;
pub const count_leading_zeros = UnaryFn("stablehlo.count_leading_zeros").call;
pub const is_finite = UnaryFn("stablehlo.is_finite").call;
pub const logistic = UnaryFn("stablehlo.logistic").call;
pub const popcnt = UnaryFn("stablehlo.popcnt").call;
pub const sign = UnaryFn("stablehlo.sign").call;
pub const real = UnaryFn("stablehlo.real").call;
pub const imag = UnaryFn("stablehlo.imag").call;

pub const add = BinaryFn("stablehlo.add").call;
pub const multiply = BinaryFn("stablehlo.multiply").call;
pub const divide = BinaryFn("stablehlo.divide").call;
pub const subtract = BinaryFn("stablehlo.subtract").call;
pub const or_ = BinaryFn("stablehlo.or").call;
pub const xor = BinaryFn("stablehlo.xor").call;
pub const and_ = BinaryFn("stablehlo.and").call;
pub const atan2 = BinaryFn("stablehlo.atan2").call;
pub const maximum = BinaryFn("stablehlo.maximum").call;
pub const minimum = BinaryFn("stablehlo.minimum").call;
pub const power = BinaryFn("stablehlo.power").call;
pub const remainder = BinaryFn("stablehlo.remainder").call;
pub const shift_left = BinaryFn("stablehlo.shift_left").call;
pub const shift_right_arithmetic = BinaryFn("stablehlo.shift_right_arithmetic").call;
pub const shift_right_logical = BinaryFn("stablehlo.shift_right_logical").call;
pub const complex = BinaryFn("stablehlo.complex").call;

fn UnaryFn(comptime op_name: [:0]const u8) type {
    return struct {
        pub fn call(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = &.{value},
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

pub fn BinaryFn(comptime op_name: [:0]const u8) type {
    return struct {
        pub fn call(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, location: mlir.Location) mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = &.{ lhs, rhs },
                .result_type_inference = true,
                .location = location,
            });
        }
    };
}

pub fn return_(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .variadic_operands = &.{&.{value}},
        .verify = false,
        .location = location,
    });
}

pub fn returns_(ctx: mlir.Context, values: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.return", .{
        .variadic_operands = &.{values},
        .verify = false,
        .location = location,
    });
}

pub fn bitcast_convert(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.bitcast_convert", .{
        .operands = &.{value},
        .results = &.{result_type},
        .location = location,
    });
}

pub fn cholesky(ctx: mlir.Context, value: mlir.Value, lower: bool, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.cholesky", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "lower", .i1_from_bool(ctx, lower) },
        },
        .location = location,
    });
}

pub fn clamp(ctx: mlir.Context, min: mlir.Value, value: mlir.Value, max: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.clamp", .{
        .operands = &.{ min, value, max },
        .result_type_inference = true,
        .location = location,
    });
}

pub const DotPrecision = union(enum) {
    fast,
    high,
    highest,
    algorithm: DotAlgorithm,

    pub fn precision_attr(self: DotPrecision, ctx: mlir.Context) mlir.Attribute {
        const precision = PrecisionAttribute.init(ctx, switch (self) {
            .fast => .DEFAULT,
            .high => .HIGH,
            .highest => .HIGHEST,
            // The algorithm attribute supplies its own precision behavior.
            .algorithm => .DEFAULT,
        });
        return precision.as_attr();
    }

    pub fn algorithm_attr(self: DotPrecision, ctx: mlir.Context, operand_type: mlir.RankedTensorType) ?mlir.Attribute {
        return switch (self) {
            .algorithm => |algo| algo.as_attr(ctx, operand_type),
            else => null,
        };
    }
};

pub const DotAlgorithm = struct {
    accumulation: mlir.FloatTypes,
    // StableHLO accepts separate component counts for each operand.
    //
    // Zigrad currently represents algorithms with equal component counts.
    component_count: u8 = 1,
    num_primitive_operations: u8 = 1,
    allow_imprecise_accumulation: bool = false,

    // TODO(stablehlo): Add a validated BF16 six-operation preset.

    pub fn as_attr(self: DotAlgorithm, ctx: mlir.Context, tensor_type: mlir.RankedTensorType) mlir.Attribute {
        const elem_type = tensor_type.get_element_type();

        return mlir.Attribute.wrap(c.stablehloDotAlgorithmGet(
            ctx._inner,
            elem_type._inner,
            elem_type._inner,
            self.accumulation.as_type(ctx)._inner,
            self.component_count,
            self.component_count,
            self.num_primitive_operations,
            self.allow_imprecise_accumulation,
        ));
    }
};

/// Creates a StableHLO `dot_general` operation with an explicit result type.
pub fn dot_general(
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    result_type: mlir.Type,
    location: mlir.Location,
    opts: struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        precision: DotPrecision,
    },
) mlir.Operation {
    const precisions: [2]mlir.Attribute = @splat(opts.precision.precision_attr(ctx));
    const attributes = [3]mlir.AttrTuple{
        .{
            "dot_dimension_numbers",
            DotDimensionNumbersAttribute.init(ctx, .{
                .lhs_batching_dimensions = opts.lhs_batching_dimensions,
                .rhs_batching_dimensions = opts.rhs_batching_dimensions,
                .lhs_contracting_dimensions = opts.lhs_contracting_dimensions,
                .rhs_contracting_dimensions = opts.rhs_contracting_dimensions,
            }).as_attr(),
        },
        .{ "precision_config", .array(ctx, &precisions) },
        // Keeping the optional attribute last makes the prefix slice valid.
        .{ "algorithm", opts.precision.algorithm_attr(ctx, lhs.get_type().as(mlir.RankedTensorType).?) orelse undefined },
    };
    const n_attributes = if (opts.precision == .algorithm) attributes.len else attributes.len - 1;
    return mlir.Operation.make(ctx, "stablehlo.dot_general", .{
        .operands = &.{ lhs, rhs },
        .results = &.{result_type},
        .attributes = attributes[0..n_attributes],
        .location = location,
    });
}

pub fn constant(
    ctx: mlir.Context,
    dims: []const i64,
    elem_type: mlir.DenseElementsAttributeTypes,
    raw_bytes: []const u8,
    location: mlir.Location,
) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.constant", .{
        .operands = &.{},
        .results = &.{.tensor(dims, elem_type.mlir_type(ctx))},
        .attributes = &.{.{ "value", .dense_elements_from_bytes(ctx, dims, elem_type, raw_bytes) }},
        .location = location,
    });
}

pub fn convert(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.convert", .{
        .operands = &.{value},
        .results = &.{result_type},
        .location = location,
    });
}

pub fn broadcast_in_dim(ctx: mlir.Context, operand: mlir.Value, dims: []const i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.broadcast_in_dim", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "broadcast_dimensions", .dense(ctx, .i64, dims) },
        },
        .location = location,
    });
}

pub fn transpose(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location, opts: struct { permutation: []const i64 }) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.transpose", .{
        .operands = &.{value},
        .results = &.{result_type},
        .attributes = &.{
            .{ "permutation", .dense(ctx, .i64, opts.permutation) },
        },
        .location = location,
    });
}

pub fn slice(ctx: mlir.Context, operand: mlir.Value, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.slice", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "start_indices", .dense(ctx, .i64, start_indices) },
            .{ "limit_indices", .dense(ctx, .i64, limit_indices) },
            .{ "strides", .dense(ctx, .i64, strides) },
        },
        .location = location,
    });
}

pub fn concatenate(ctx: mlir.Context, inputs: []const mlir.Value, dimension: i64, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.concatenate", .{
        .operands = inputs,
        .result_type_inference = true,
        .attributes = &.{
            .{ "dimension", .int(ctx, .i64, dimension) },
        },
        .location = location,
    });
}

pub fn reshape(ctx: mlir.Context, value: mlir.Value, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reshape", .{
        .operands = &.{value},
        .results = &.{result_type},
        .location = location,
    });
}

pub fn select(ctx: mlir.Context, condition: mlir.Value, then: mlir.Value, else_: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.select", .{
        .operands = &.{ condition, then, else_ },
        .results = &.{then.get_type()},
        .location = location,
    });
}

pub fn gather(
    ctx: mlir.Context,
    value: mlir.Value,
    indices: mlir.Value,
    slice_sizes: []const i64,
    location: mlir.Location,
    args: struct {
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        indices_are_sorted: bool = false,
    },
) mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.gather",
        .{
            .operands = &.{ value, indices },
            .result_type_inference = true,
            .attributes = &.{
                .{ "dimension_numbers", GatherDimensionNumbersAttribute.init(
                    ctx,
                    args.offset_dims,
                    args.collapsed_slice_dims,
                    args.operand_batching_dims,
                    args.start_indices_batching_dims,
                    args.start_index_map,
                    args.index_vector_dim,
                ).as_attr() },
                .{ "slice_sizes", .dense(ctx, .i64, slice_sizes) },
                .{ "indices_are_sorted", .boolean(ctx, args.indices_are_sorted) },
            },
            .location = location,
        },
    );
}

fn element_type_or_self(typ: mlir.Type) mlir.Type {
    return if (typ.as(mlir.RankedTensorType)) |shaped| {
        return shaped.get_element_type();
    } else typ;
}

pub const ScatterArgs = struct {
    update_window_dims: []const i64,
    inserted_window_dims: []const i64,
    input_batching_dims: []const i64,
    scatter_indices_batching_dims: []const i64,
    scatter_dims_to_operand_dims: []const i64,
    index_vector_dim: i64,
    indices_are_sorted: bool = false,
    unique_indices: bool = false,

    pub fn get_scatter_dimension_numbers(self: ScatterArgs, ctx: mlir.Context) mlir.Attribute {
        return .{ ._inner = c.stablehloScatterDimensionNumbersGet(
            ctx._inner,
            @intCast(self.update_window_dims.len),
            self.update_window_dims.ptr,
            @intCast(self.inserted_window_dims.len),
            self.inserted_window_dims.ptr,
            @intCast(self.input_batching_dims.len),
            self.input_batching_dims.ptr,
            @intCast(self.scatter_indices_batching_dims.len),
            self.scatter_indices_batching_dims.ptr,
            @intCast(self.scatter_dims_to_operand_dims.len),
            self.scatter_dims_to_operand_dims.ptr,
            self.index_vector_dim,
        ) };
    }
};

pub fn scatter(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    scatter_indices: []const mlir.Value,
    updates: []const mlir.Value,
    update_block: mlir.Block,
    args: ScatterArgs,
    location: mlir.Location,
) mlir.Operation {
    return mlir.Operation.make(
        ctx,
        "stablehlo.scatter",
        .{
            .variadic_operands = &.{ inputs, scatter_indices, updates },
            .blocks = &.{update_block},
            .attributes = &.{
                .{ "scatter_dimension_numbers", args.get_scatter_dimension_numbers(ctx) },
                .{ "indices_are_sorted", .boolean(ctx, args.indices_are_sorted) },
                .{ "unique_indices", .boolean(ctx, args.unique_indices) },
            },
            .result_type_inference = true,
            .location = location,
        },
    );
}

pub fn iota(ctx: mlir.Context, dimension: i64, result_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.iota", .{
        .operands = &.{},
        .results = &.{result_type},
        .attributes = &.{
            .{ "iota_dimension", .int(ctx, .i64, dimension) },
        },
        .location = location,
    });
}

pub fn reverse(ctx: mlir.Context, operand: mlir.Value, dimensions: []const i64, location: mlir.Location) mlir.Operation {
    const result_type = operand.get_type();
    return mlir.Operation.make(ctx, "stablehlo.reverse", .{
        .operands = &.{operand},
        .results = &.{result_type},
        .attributes = &.{
            .{ "dimensions", .dense(ctx, .i64, dimensions) },
        },
        .location = location,
    });
}

pub fn compare(ctx: mlir.Context, lhs: mlir.Value, rhs: mlir.Value, comparison_direction: ComparisonDirection, compare_type: CompareType, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.compare", .{
        .operands = &.{ lhs, rhs },
        .result_type_inference = true,
        .attributes = &.{
            .{ "comparison_direction", comparison_direction.as_attr() },
            .{ "compare_type", compare_type.as_attr() },
        },
        .location = location,
    });
}

pub fn reduce(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    init_values: []const mlir.Value,
    dimensions: []const i64,
    blkctx: anytype,
    blkfn: fn (anytype, mlir.Context, []const mlir.Value, []const mlir.Value) mlir.Operation,
    location: mlir.Location,
) mlir.Operation {
    const MaxBlockArguments = 32;

    const block_n_args = inputs.len + init_values.len;
    const locations = (@as([MaxBlockArguments]mlir.Location, @splat(mlir.Location.unknown(ctx))))[0..block_n_args];
    var reduce_elem_types: [MaxBlockArguments]mlir.Type = undefined;
    for (inputs, 0..) |input, i| {
        const arg_type: mlir.Type = .tensor(&.{}, element_type_or_self(input.get_type()));
        reduce_elem_types[i] = arg_type;
        reduce_elem_types[inputs.len + i] = arg_type;
    }
    var block = mlir.Block.init(reduce_elem_types[0..block_n_args], locations) catch unreachable;

    var block_inputs: [MaxBlockArguments / 2]mlir.Value = undefined;
    var block_accs: [MaxBlockArguments / 2]mlir.Value = undefined;
    for (0..inputs.len) |i| {
        block_inputs[i] = block.argument(i);
        block_accs[i] = block.argument(inputs.len + i);
    }
    const reduce_op = blkfn(blkctx, ctx, block_inputs[0..inputs.len], block_accs[0..init_values.len]);
    block.append_operation(reduce_op);
    const ret = return_(ctx, reduce_op.result(0), mlir.Location.unknown(ctx));
    block.append_operation(ret);

    return mlir.Operation.make(ctx, "stablehlo.reduce", .{
        .variadic_operands = &.{ inputs, init_values },
        .result_type_inference = true,
        .blocks = &.{block},
        .attributes = &.{
            .{ "dimensions", .dense(ctx, .i64, dimensions) },
        },
        .location = location,
    });
}

pub fn sort(
    ctx: mlir.Context,
    inputs: []const mlir.Value,
    dimension: i64,
    is_stable: bool,
    blkctx: anytype,
    compfn: fn (anytype, mlir.Context, []const mlir.Value) mlir.Operation,
    location: mlir.Location,
) mlir.Operation {
    const MaxBlockArguments = 32;

    const locations = (@as([MaxBlockArguments]mlir.Location, @splat(mlir.Location.unknown(ctx))))[0 .. inputs.len * 2];
    var sort_elem_types: [MaxBlockArguments]mlir.Type = undefined;
    for (inputs, 0..) |input, i| {
        const arg_type: mlir.Type = .tensor(&.{}, element_type_or_self(input.get_type()));
        sort_elem_types[i * 2] = arg_type;
        sort_elem_types[i * 2 + 1] = arg_type;
    }
    var block = mlir.Block.init(sort_elem_types[0 .. inputs.len * 2], locations) catch unreachable;

    var block_inputs: [MaxBlockArguments]mlir.Value = undefined;
    for (0..inputs.len * 2) |i| {
        block_inputs[i] = block.argument(i);
    }
    _ = compfn(blkctx, ctx, block_inputs[0 .. inputs.len * 2]);

    return mlir.Operation.make(ctx, "stablehlo.sort", .{
        .variadic_operands = &.{inputs},
        .result_type_inference = true,
        .blocks = &.{block},
        .attributes = &.{
            .{ "dimension", .int(ctx, .i64, dimension) },
            .{ "is_stable", .boolean(ctx, is_stable) },
        },
        .location = location,
    });
}

pub fn dynamic_slice(ctx: mlir.Context, operand: mlir.Value, new_dims: []const i64, start_indices: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_slice", .{
        .variadic_operands = &.{ &.{operand}, start_indices },
        .result_type_inference = true,
        .attributes = &.{
            .{ "slice_sizes", .dense(ctx, .i64, new_dims) },
        },
        .location = location,
    });
}

pub fn round_nearest_afz(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.round_nearest_afz", .{
        .operands = &.{value},
        .result_type_inference = true,
        .location = location,
    });
}

pub fn round_nearest_even(ctx: mlir.Context, value: mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.round_nearest_even", .{
        .operands = &.{value},
        .result_type_inference = true,
        .location = location,
    });
}

pub const PadOpts = struct {
    low: []const i64,
    high: []const i64,
    interior: []const i64,
};

pub fn pad(ctx: mlir.Context, value: mlir.Value, padding_value: mlir.Value, opts: PadOpts, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.pad", .{
        .operands = &.{ value, padding_value },
        .result_type_inference = true,
        .attributes = &.{
            .{ "edge_padding_low", .dense(ctx, .i64, opts.low) },
            .{ "edge_padding_high", .dense(ctx, .i64, opts.high) },
            .{ "interior_padding", .dense(ctx, .i64, opts.interior) },
        },
        .location = location,
    });
}

pub const TriangularSolveOpts = struct {
    left_side: bool,
    lower: bool,
    unit_diagonal: bool,
    transpose_a: Transpose.Type,
};

pub fn triangular_solve(ctx: mlir.Context, value: mlir.Value, other: mlir.Value, location: mlir.Location, opts: TriangularSolveOpts) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.triangular_solve", .{
        .operands = &.{ value, other },
        .result_type_inference = true,
        .attributes = &.{
            .{ "left_side", .i1_from_bool(ctx, opts.left_side) },
            .{ "lower", .i1_from_bool(ctx, opts.lower) },
            .{ "unit_diagonal", .i1_from_bool(ctx, opts.unit_diagonal) },
            .{ "transpose_a", Transpose.init(ctx, opts.transpose_a).as_attr() },
        },
        .location = location,
    });
}

pub const FftOpts = struct {
    kind: FftType.Type,
    length: []const i64,
};

pub fn fft(ctx: mlir.Context, value: mlir.Value, location: mlir.Location, opts: FftOpts) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.fft", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "fft_type", FftType.init(ctx, opts.kind).as_attr() },
            .{ "fft_length", .dense(ctx, .i64, opts.length) },
        },
        .location = location,
    });
}

pub fn rng(ctx: mlir.Context, a: mlir.Value, b: mlir.Value, shape: mlir.Value, rng_distribution: RngDistribution.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng", .{
        .operands = &.{ a, b, shape },
        .result_type_inference = true,
        .attributes = &.{
            .{ "rng_distribution", RngDistribution.init(ctx, rng_distribution).as_attr() },
        },
        .location = location,
    });
}

pub fn rng_bit_generator(ctx: mlir.Context, rng_algorithm: RngAlgorithm.Type, initial_state: mlir.Value, res_state_type: mlir.Type, res_type: mlir.Type, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.rng_bit_generator", .{
        .operands = &.{initial_state},
        .results = &.{ res_state_type, res_type },
        .attributes = &.{
            .{ "rng_algorithm", RngAlgorithm.init(ctx, rng_algorithm).as_attr() },
        },
        .location = location,
    });
}

pub fn reduce_precision(ctx: mlir.Context, value: mlir.Value, exponent_bits: i32, mantissa_bits: i32, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.reduce_precision", .{
        .operands = &.{value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "exponent_bits", .int(ctx, .i32, exponent_bits) },
            .{ "mantissa_bits", .int(ctx, .i32, mantissa_bits) },
        },
        .location = location,
    });
}

pub fn dynamic_update_slice(ctx: mlir.Context, operand: mlir.Value, update: mlir.Value, start_indices: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.dynamic_update_slice", .{
        .variadic_operands = &.{ &.{operand}, &.{update}, start_indices },
        .result_type_inference = true,
        .location = location,
    });
}

pub fn tuple(ctx: mlir.Context, values: []const mlir.Value, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.tuple", .{
        .operands = values,
        .result_type_inference = true,
        .location = location,
    });
}

pub fn get_tuple_element(ctx: mlir.Context, tuple_value: mlir.Value, index: i64, location: mlir.Location) mlir.Operation {
    return mlir.Operation.make(ctx, "stablehlo.get_tuple_element", .{
        .operands = &.{tuple_value},
        .result_type_inference = true,
        .attributes = &.{
            .{ "index", .int(ctx, .i32, index) },
        },
        .location = location,
    });
}

pub const ConvolutionOpts = struct {
    window_strides: []const i64,
    pad_value: []const i64,
    pad_shape: []const i64 = &.{},
    lhs_dilation: []const i64,
    rhs_dilation: []const i64,
    window_reversal: []const bool,
    input_batch_dimension: i64,
    input_feature_dimension: i64,
    input_spatial_dimensions: []const i64,
    kernel_input_feature_dimension: i64,
    kernel_output_feature_dimension: i64,
    kernel_spatial_dimensions: []const i64,
    output_batch_dimension: i64,
    output_feature_dimension: i64,
    output_spatial_dimensions: []const i64,
    feature_group_count: i64,
    batch_group_count: i64,
    /// Precision settings for the convolution's two operands.
    precision_config: []const PrecisionAttribute.Precision = &.{},
};

pub fn convolution(
    ctx: mlir.Context,
    lhs: mlir.Value,
    rhs: mlir.Value,
    opts: ConvolutionOpts,
    res_type: mlir.Type,
    location: mlir.Location,
) mlir.Operation {
    var max_precisions: [2]mlir.Attribute = undefined;
    if (opts.precision_config.len > max_precisions.len)
        @panic("stablehlo convolution accepts at most two precision settings");
    for (opts.precision_config, 0..) |p, i| {
        max_precisions[i] = PrecisionAttribute.init(ctx, p).as_attr();
    }
    var window_reversal: [3]i32 = undefined;
    for (opts.window_reversal, 0..) |w, i| {
        window_reversal[i] = @intCast(@intFromBool(w));
    }
    return mlir.Operation.make(ctx, "stablehlo.convolution", .{
        .operands = &.{ lhs, rhs },
        .results = &.{res_type},
        .attributes = &.{
            .{ "window_strides", .dense(ctx, .i64, opts.window_strides) },
            .{ "padding", .dense_elements(ctx, opts.pad_shape, .i64, opts.pad_value) },
            .{ "lhs_dilation", .dense(ctx, .i64, opts.lhs_dilation) },
            .{ "rhs_dilation", .dense(ctx, .i64, opts.rhs_dilation) },
            .{ "window_reversal", .dense(ctx, .bool, window_reversal[0..opts.window_reversal.len]) },
            .{
                "dimension_numbers",
                ConvDimensionNumbersAttribute.init(ctx, .{
                    .input_batch_dimension = opts.input_batch_dimension,
                    .input_feature_dimension = opts.input_feature_dimension,
                    .input_spatial_dimensions = opts.input_spatial_dimensions,
                    .kernel_input_feature_dimension = opts.kernel_input_feature_dimension,
                    .kernel_output_feature_dimension = opts.kernel_output_feature_dimension,
                    .kernel_spatial_dimensions = opts.kernel_spatial_dimensions,
                    .output_batch_dimension = opts.output_batch_dimension,
                    .output_feature_dimension = opts.output_feature_dimension,
                    .output_spatial_dimensions = opts.output_spatial_dimensions,
                }).as_attr(),
            },
            .{ "feature_group_count", .int(ctx, .i64, opts.feature_group_count) },
            .{ "batch_group_count", .int(ctx, .i64, opts.batch_group_count) },
            .{ "precision_config", .array(ctx, max_precisions[0..opts.precision_config.len]) },
        },
        .location = location,
    });
}

pub const CustomCallOpts = struct {
    pub const ApiVersion = enum(i32) {
        original = 1,
        status_returning = 2,
        status_returning_unified = 3,
        typed_ffi = 4,
    };

    call_target_name: []const u8,
    has_side_effect: bool,
    backend_config: ?mlir.Attribute,
    operand_layouts: ?[]const []const usize = null,
    result_layouts: ?[]const []const usize = null,
    output_operand_aliases: []const i64 = &.{},
    additional_attributes: []const mlir.AttrTuple = &.{},
    api_version: ApiVersion,
};

pub fn custom_call(ctx: mlir.Context, inputs: []const mlir.Value, opts: CustomCallOpts, res_types: []const mlir.Type, location: mlir.Location) mlir.Operation {
    const MAX_OPERANDS = 64;
    const MAX_RESULTS = 16;

    const backend_config = opts.backend_config orelse mlir.Attribute.string(ctx, "");
    if (@backingInt(opts.api_version) < @backingInt(CustomCallOpts.ApiVersion.typed_ffi)) {
        std.debug.assert(backend_config.is_a(mlir.StringAttribute));
    } else {
        std.debug.assert(backend_config.is_a(mlir.DictionaryAttribute));
    }

    var attrs = BoundedArray(mlir.AttrTuple, 32){};
    attrs.append_slice_assume_capacity(&[_]mlir.AttrTuple{
        .{ "api_version", .int(ctx, .i32, @backingInt(opts.api_version)) },
        .{ "call_target_name", .string(ctx, opts.call_target_name) },
        .{ "has_side_effect", .boolean(ctx, opts.has_side_effect) },
        .{ "backend_config", backend_config },
    });

    {
        var output_operand_aliases = BoundedArray(mlir.Attribute, MAX_RESULTS){};
        for (opts.output_operand_aliases) |alias| {
            output_operand_aliases.append_assume_capacity(
                OutputOperandAliasAttribute.init(ctx, &.{}, alias, &.{}).as_attr(),
            );
        }
        attrs.append_assume_capacity(.{ "output_operand_aliases", .array(ctx, output_operand_aliases.const_slice()) });
    }

    const MINOR_TO_MAJOR = blk: {
        const MAX_RANK = 8;
        var ret: [MAX_RANK]usize = undefined;
        for (0..MAX_RANK) |i| {
            ret[i] = @intCast(MAX_RANK - i - 1);
        }
        break :blk ret;
    };

    if (opts.operand_layouts) |layouts| {
        var operand_layouts = BoundedArray(mlir.Attribute, MAX_OPERANDS){};
        for (layouts) |ol| {
            operand_layouts.append_assume_capacity(.dense_elements(ctx, &.{@intCast(ol.len)}, .index, ol));
        }
        attrs.append_assume_capacity(.{ "operand_layouts", .array(ctx, operand_layouts.const_slice()) });
    } else {
        const operand_layouts = blk: {
            var ret = BoundedArray(mlir.Attribute, MAX_OPERANDS){};
            for (inputs) |input| {
                const ranked_type = input.get_type().as(mlir.RankedTensorType).?;
                const ol = MINOR_TO_MAJOR[MINOR_TO_MAJOR.len - ranked_type.get_rank() ..];
                ret.append_assume_capacity(.dense_elements(ctx, &.{@intCast(ol.len)}, .index, ol));
            }
            break :blk ret;
        };
        attrs.append_assume_capacity(.{ "operand_layouts", .array(ctx, operand_layouts.const_slice()) });
    }

    if (opts.result_layouts) |layouts| {
        var result_layouts = BoundedArray(mlir.Attribute, MAX_RESULTS){};
        for (layouts) |rl| {
            result_layouts.append_assume_capacity(.dense_elements(ctx, &.{@intCast(rl.len)}, .index, rl));
        }
        attrs.append_assume_capacity(.{ "result_layouts", .array(ctx, result_layouts.const_slice()) });
    } else {
        const result_layouts = blk: {
            var ret = BoundedArray(mlir.Attribute, MAX_RESULTS){};
            for (res_types) |t| {
                const ranked_t = t.as(mlir.RankedTensorType).?;
                const rl = MINOR_TO_MAJOR[MINOR_TO_MAJOR.len - ranked_t.get_rank() ..];
                ret.append_assume_capacity(.dense_elements(ctx, &.{@intCast(rl.len)}, .index, rl));
            }
            break :blk ret;
        };
        attrs.append_assume_capacity(.{ "result_layouts", .array(ctx, result_layouts.const_slice()) });
    }

    attrs.append_slice(opts.additional_attributes) catch @panic("Too many additional_attributes");

    return mlir.Operation.make(ctx, "stablehlo.custom_call", .{
        .operands = inputs,
        .results = res_types,
        .attributes = attrs.const_slice(),
        .location = location,
    });
}

pub const DotDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsADotDimensionNumbers;
    const Self = DotDimensionNumbersAttribute;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub fn init(ctx: mlir.Context, args: struct {
        lhs_batching_dimensions: []const i64,
        rhs_batching_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
    }) Self {
        return .{
            ._inner = c.stablehloDotDimensionNumbersGet(
                ctx._inner,
                @intCast(args.lhs_batching_dimensions.len),
                args.lhs_batching_dimensions.ptr,
                @intCast(args.rhs_batching_dimensions.len),
                args.rhs_batching_dimensions.ptr,
                @intCast(args.lhs_contracting_dimensions.len),
                args.lhs_contracting_dimensions.ptr,
                @intCast(args.rhs_contracting_dimensions.len),
                args.rhs_contracting_dimensions.ptr,
            ),
        };
    }

    pub fn get_lhs_batching_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(self._inner));
    }

    pub fn get_lhs_batching_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(self._inner, @intCast(pos));
    }

    pub fn get_rhs_batching_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(self._inner));
    }

    pub fn get_rhs_batching_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(self._inner, @intCast(pos));
    }

    pub fn get_lhs_contracting_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(self._inner));
    }

    pub fn get_lhs_contracting_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(self._inner, @intCast(pos));
    }

    pub fn get_rhs_contracting_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(self._inner));
    }

    pub fn get_rhs_contracting_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(self._inner, @intCast(pos));
    }
};

pub const GatherDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAGatherDimensionNumbers;
    const Self = GatherDimensionNumbersAttribute;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub fn init(
        ctx: mlir.Context,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
    ) Self {
        return .{
            ._inner = c.stablehloGatherDimensionNumbersGet(
                ctx._inner,
                @intCast(offset_dims.len),
                offset_dims.ptr,
                @intCast(collapsed_slice_dims.len),
                collapsed_slice_dims.ptr,
                @intCast(operand_batching_dims.len),
                operand_batching_dims.ptr,
                @intCast(start_indices_batching_dims.len),
                start_indices_batching_dims.ptr,
                @intCast(start_index_map.len),
                start_index_map.ptr,
                index_vector_dim,
            ),
        };
    }

    pub fn get_offset_dims_size(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOffsetDimsSize(self._inner));
    }

    pub fn get_offset_dims_elem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOffsetDimsElem(self._inner, @intCast(pos));
    }

    pub fn get_collapsed_slice_dims_size(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(self._inner));
    }

    pub fn get_collapsed_slice_dims_elem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(self._inner, @intCast(pos));
    }

    pub fn get_start_index_map_size(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndexMapSize(self._inner));
    }

    pub fn get_operand_batching_dims_size(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(self._inner));
    }

    pub fn get_operand_batching_dims_elem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(self._inner, @intCast(pos));
    }

    pub fn get_start_indices_batching_dims_size(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(self._inner));
    }

    pub fn get_start_indices_batching_dims_elem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(self._inner, @intCast(pos));
    }

    pub fn get_start_index_map_elem(self: Self, pos: usize) i64 {
        return c.stablehloGatherDimensionNumbersGetStartIndexMapElem(self._inner, @intCast(pos));
    }

    pub fn get_index_vector_dim(self: Self) usize {
        return @intCast(c.stablehloGatherDimensionNumbersGetIndexVectorDim(self._inner));
    }
};

pub const ConvDimensionNumbersAttribute = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAConvDimensionNumbers;
    const Self = ConvDimensionNumbersAttribute;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub fn init(ctx: mlir.Context, args: struct {
        input_batch_dimension: i64,
        input_feature_dimension: i64,
        input_spatial_dimensions: []const i64,
        kernel_input_feature_dimension: i64,
        kernel_output_feature_dimension: i64,
        kernel_spatial_dimensions: []const i64,
        output_batch_dimension: i64,
        output_feature_dimension: i64,
        output_spatial_dimensions: []const i64,
    }) Self {
        return .{
            ._inner = c.stablehloConvDimensionNumbersGet(
                ctx._inner,
                args.input_batch_dimension,
                args.input_feature_dimension,
                @intCast(args.input_spatial_dimensions.len),
                args.input_spatial_dimensions.ptr,
                args.kernel_input_feature_dimension,
                args.kernel_output_feature_dimension,
                @intCast(args.kernel_spatial_dimensions.len),
                args.kernel_spatial_dimensions.ptr,
                args.output_batch_dimension,
                args.output_feature_dimension,
                @intCast(args.output_spatial_dimensions.len),
                args.output_spatial_dimensions.ptr,
            ),
        };
    }

    pub fn get_input_batch_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetInputBatchDimension(self._inner);
    }

    pub fn get_input_feature_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetInputFeatureDimension(self._inner);
    }

    pub fn get_input_spatial_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(self._inner));
    }

    pub fn get_input_spatial_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(self._inner, @intCast(pos));
    }

    pub fn get_kernel_input_feature_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(self._inner);
    }

    pub fn get_kernel_output_feature_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(self._inner);
    }

    pub fn get_kernel_spatial_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(self._inner));
    }

    pub fn get_kernel_spatial_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(self._inner, @intCast(pos));
    }

    pub fn get_output_batch_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetOutputBatchDimension(self._inner);
    }

    pub fn get_output_feature_dimension(self: Self) i64 {
        return c.stablehloConvDimensionNumbersGetOutputFeatureDimension(self._inner);
    }

    pub fn get_output_spatial_dimensions_size(self: Self) usize {
        return @intCast(c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(self._inner));
    }

    pub fn get_output_spatial_dimensions_elem(self: Self, pos: usize) i64 {
        return c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(self._inner, @intCast(pos));
    }
};

pub const OutputOperandAliasAttribute = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAOutputOperandAlias;
    pub const as_attr = mlir.Attribute.from_any(OutputOperandAliasAttribute);
    pub const eql = mlir.Attribute.eql_any(OutputOperandAliasAttribute);

    pub fn init(
        ctx: mlir.Context,
        output_tuple_indices: []const i64,
        operand_index: i64,
        operand_tuple_indices: []const i64,
    ) OutputOperandAliasAttribute {
        return .{ ._inner = c.stablehloOutputOperandAliasGet(
            ctx._inner,
            @intCast(output_tuple_indices.len),
            output_tuple_indices.ptr,
            @intCast(operand_index),
            @intCast(operand_tuple_indices.len),
            operand_tuple_indices.ptr,
        ) };
    }
};

pub const PrecisionAttribute = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAPrecisionAttr;
    const Self = PrecisionAttribute;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Precision = enum {
        DEFAULT,
        HIGH,
        HIGHEST,
    };

    pub fn init(ctx: mlir.Context, value: Precision) Self {
        return .{ ._inner = c.stablehloPrecisionAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Precision {
        const value = mlir.from_string_ref(c.stablehloPrecisionAttrGetValue(self._inner));
        return std.meta.stringToEnum(Precision, value) orelse unreachable;
    }
};

pub const ComparisonDirection = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAComparisonDirectionAttr;
    const Self = ComparisonDirection;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Direction = enum {
        EQ,
        NE,
        GE,
        GT,
        LE,
        LT,
    };

    pub fn init(ctx: mlir.Context, value: Direction) Self {
        return .{ ._inner = c.stablehloComparisonDirectionAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Direction {
        const value = mlir.from_string_ref(c.stablehloComparisonDirectionAttrGetValue(self._inner));
        return std.meta.stringToEnum(Direction, value) orelse unreachable;
    }
};

pub const CompareType = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAComparisonTypeAttr;
    const Self = CompareType;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Type = enum {
        SIGNED,
        UNSIGNED,
        FLOAT,
        TOTALORDER,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return .{ ._inner = c.stablehloComparisonTypeAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Type {
        const value = mlir.from_string_ref(c.stablehloComparisonTypeAttrGetValue(self._inner));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const Transpose = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsATransposeAttr;
    const Self = Transpose;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Type = enum {
        NO_TRANSPOSE,
        TRANSPOSE,
        ADJOINT,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return .{ ._inner = c.stablehloTransposeAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Type {
        const value = mlir.from_string_ref(c.stablehloTransposeAttrGetValue(self._inner));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const FftType = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsAFftTypeAttr;
    const Self = FftType;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Type = enum {
        FFT,
        IFFT,
        RFFT,
        IRFFT,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return .{ ._inner = c.stablehloFftTypeAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Type {
        const value = mlir.from_string_ref(c.stablehloFftTypeAttrGetValue(self._inner));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const RngDistribution = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsARngDistributionAttr;
    const Self = RngDistribution;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Type = enum {
        UNIFORM,
        NORMAL,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return .{ ._inner = c.stablehloRngDistributionAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Type {
        const value = mlir.from_string_ref(c.stablehloRngDistributionAttrGetValue(self._inner));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub const RngAlgorithm = struct {
    _inner: c.MlirAttribute,

    pub const is_a_fn = c.stablehloAttributeIsARngAlgorithmAttr;
    const Self = RngAlgorithm;
    pub const as_attr = mlir.Attribute.from_any(Self);
    pub const eql = mlir.Attribute.eql_any(Self);

    pub const Type = enum {
        DEFAULT,
        THREE_FRY,
        PHILOX,
    };

    pub fn init(ctx: mlir.Context, value: Type) Self {
        return .{ ._inner = c.stablehloRngAlgorithmAttrGet(ctx._inner, mlir.string_ref(@tagName(value))) };
    }

    pub fn get_value(self: Self) Type {
        const value = mlir.from_string_ref(c.stablehloRngAlgorithmAttrGetValue(self._inner));
        return std.meta.stringToEnum(Type, value) orelse unreachable;
    }
};

pub fn stablehlo_version_from_compatibility_requirement(requirement: c.MlirStablehloCompatibilityRequirement) []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;

        fn call(req: c.MlirStablehloCompatibilityRequirement) []u8 {
            var w: std.Io.Writer = .fixed(&buf);
            var context = .{ .writer = &w };
            const WriterContext = @TypeOf(context);

            c.stablehlo_version_from_compatibility_requirement(req, (struct {
                pub fn callback(mlir_str: c.MlirStringRef, userdata: ?*anyopaque) callconv(.c) void {
                    const inner_ctx: *WriterContext = @ptrCast(@alignCast(userdata));
                    inner_ctx.writer.writeAll(mlir.from_string_ref(mlir_str)) catch unreachable;
                }
            }).callback, &context);

            return w.buffered();
        }
    };

    return state.call(requirement);
}

pub fn stablehlo_get_smaller_version(version1: []const u8, version2: []const u8) []const u8 {
    const Cmp = struct {
        v1: []const u8,
        v1_is_smaller: bool = undefined,

        pub fn smaller_cb(smaller_version: c.MlirStringRef, opaque_cmp: ?*anyopaque) callconv(.c) void {
            var cmp: *@This() = @ptrCast(@alignCast(opaque_cmp));
            cmp.v1_is_smaller = std.mem.eql(u8, cmp.v1, smaller_version.data[0..smaller_version.length]);
        }
    };

    var cmp_ctx: Cmp = .{ .v1 = version1 };
    const cmp_res = c.stablehlo_get_smaller_version(mlir.string_ref(version1), mlir.string_ref(version2), Cmp.smaller_cb, &cmp_ctx);

    std.debug.assert(cmp_res.value != 0);
    return if (cmp_ctx.v1_is_smaller) version1 else version2;
}

pub fn get_current_version() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
        var str: []const u8 = undefined;
        var once = std.once(call);

        fn call() void {
            var writer: std.Io.Writer = .fixed(&buf);
            c.stablehloGetCurrentVersion(print_callback_no_fail, &writer);
            str = writer.buffered();
        }
    };

    state.once.call();
    return state.str;
}

pub fn get_minimum_version() []const u8 {
    const state = struct {
        var buf: [32]u8 = undefined;
        var str: []const u8 = undefined;
        var once = std.once(call);

        fn call() void {
            var writer: std.Io.Writer = .fixed(&buf);
            c.stablehloGetMinimumVersion(print_callback_no_fail, &writer);
            str = writer.buffered();
        }
    };

    state.once.call();
    return state.str;
}

pub fn serialize_portable_artifact(
    bytecode: []const u8,
    target_version: []const u8,
    writer: *std.Io.Writer,
) error{ InvalidMlirBytecodeVersion, WriteFailed }!void {
    var writer_err: mlir.WriterWithErr = .{ .writer = writer };
    try mlir.success_or(
        c.stablehloSerializePortableArtifactFromStringRef(
            mlir.string_ref(bytecode),
            mlir.string_ref(target_version),
            mlir.WriterWithErr.print_callback,
            &writer_err,
        ),
        error.InvalidMlirBytecodeVersion,
    );
    return try writer_err.check();
}

fn print_callback_no_fail(mlir_str: c.MlirStringRef, opaque_writer: ?*anyopaque) callconv(.c) void {
    const writer: *std.Io.Writer = @ptrCast(@alignCast(opaque_writer));
    writer.writeAll(mlir.from_string_ref(mlir_str)) catch @panic("Failed to write MLIR");
}
