//! PR to StableHLO lowering.
//!
//! Maps supported PR operations to StableHLO through the shared MLIR function
//!  scaffold.
//!
//! `Lower` is the compilation-composition operation. Direct functions support
//!  integration code and focused tests.
const std = @import("std");

const compilation = @import("compilation");
const pr = @import("pr");
const fingerprint = @import("pr").analysis.fingerprint;
const kernelize = @import("pr").transform.kernelize;
const outline = @import("pr").transform.outline;
const mlir = @import("../../c/mlir/mlir.zig");
const stablehlo = @import("../../c/mlir/dialects/stablehlo.zig");
const MlirSession = @import("../session.zig").Session;
const context = @import("../lowering.zig");
const log = std.log.scoped(.@"zg/lower_stablehlo");

const LowerError = context.LowerError;
const LowerContext = context.LowerContext;
const OutputFormat = context.OutputFormat;

const stablehlo_types = @import("../../stablehlo.zig");

/// Lower a PR program to StableHLO MLIR bytecode or text.
///
/// Creates an MLIR session with StableHLO registered and delegates function
///  construction to the shared MLIR scaffold.
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry: pr.FunctionId,
    out: OutputFormat,
) LowerError![]u8 {
    var session = try MlirSession.init();
    defer session.deinit();
    session.load_dialect("stablehlo");
    return try context.lower_program_to_mlir(allocator, session, program, entry, out, lower_op);
}

pub fn lower_function_to_mlir(allocator: std.mem.Allocator, func: pr.Function, out: OutputFormat) LowerError![]u8 {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    const function = program.add_function(func) catch |err| switch (err) {
        error.DuplicateFunctionName,
        error.DuplicateFunctionId,
        error.FunctionIdExhausted,
        => return error.InvalidProgram,
        error.OutOfMemory => return error.OutOfMemory,
    };
    return try lower_program_to_mlir(allocator, &program, function, out);
}

fn lower_op(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    switch (op.prim()) {
        // Binary elementwise
        .add => lower_binary(stablehlo.add, ctx, op),
        .subtract => lower_binary(stablehlo.subtract, ctx, op),
        .multiply => lower_binary(stablehlo.multiply, ctx, op),
        .divide => lower_binary(stablehlo.divide, ctx, op),
        .maximum => lower_binary(stablehlo.maximum, ctx, op),
        // Unary
        .exp => lower_unary(stablehlo.exponential, ctx, op),
        .log => lower_unary(stablehlo.log, ctx, op),
        .rsqrt => lower_unary(stablehlo.rsqrt, ctx, op),
        .logistic => lower_unary(stablehlo.logistic, ctx, op),
        // Type conversion
        .convert => try lower_convert(ctx, op),
        // Constant
        .literal => lower_literal(ctx, op),
        // Shape
        .reshape => try lower_reshape(ctx, op),
        .transpose => try lower_transpose(ctx, op),
        .broadcast_in_dim => try lower_broadcast_in_dim(ctx, op),
        .iota => lower_iota(ctx, op),
        .slice => try lower_slice(ctx, op),
        .concatenate => try lower_concatenate(ctx, op),
        // Reduction
        .reduce => try lower_reduce(ctx, op),
        // Contraction
        .dot => try lower_dot(ctx, op),
        .mm => try lower_mm(ctx, op),
        .bmm => try lower_bmm(ctx, op),
        .dot_general => try lower_dot_general(ctx, op),
        .convolution => try lower_convolution(ctx, op),
        // Compare
        .compare => try lower_compare(ctx, op),
        .select => try lower_select(ctx, op),
        // Structured
        .gather => try lower_gather(ctx, op),
        .scatter => try lower_scatter(ctx, op),
        // Special - Dialect-agnostic, delegated.
        .call => try context.lower_call(ctx, op),
        .custom_call => try lower_custom_call(ctx, op),
    }
}

/// Lower a PR custom call to StableHLO.
///
/// PJRT FFI API version 1 uses a dictionary-valued backend configuration.
///
/// TODO(stablehlo): Define aliasing and special-target policy for custom calls.
fn lower_custom_call(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    if (op.outputs.len == 0) return error.InvalidProgram;

    const cc = op.params.custom_call;

    const result_types = try ctx.arena.alloc(mlir.Type, op.outputs.len);
    for (op.outputs, 0..) |out_var, i| {
        const out_tensor = out_var.aval.as_tensor();
        result_types[i] = ctx.tensor_to_mlir_type(out_tensor);
    }

    const operand_values = try ctx.arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        operand_values[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;
    }

    var backend_fields: [1]mlir.AttrTuple = undefined;
    const backend_field_count: usize = if (cc.payload.len > 0) count: {
        backend_fields[0] = .{
            stablehlo_types.custom_call_payload_name,
            mlir.Attribute.string(ctx.mlir_ctx, cc.payload),
        };
        break :count 1;
    } else 0;
    const backend_config = mlir.Attribute.dict(
        ctx.mlir_ctx,
        backend_fields[0..backend_field_count],
    );

    const mlir_op = stablehlo.custom_call(ctx.mlir_ctx, operand_values, .{
        .call_target_name = cc.target_name,
        .has_side_effect = cc.has_side_effect,
        .backend_config = backend_config,
        // TODO(stablehlo): Select the calling convention through compiler and
        //  runtime composition.
        .api_version = .typed_ffi,
    }, result_types, ctx.loc);

    ctx.block.append_operation(mlir_op);
    for (op.outputs, 0..) |out_var, i| {
        ctx.set_value(out_var, mlir_op.result(i));
    }
}

fn lower_binary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    op: *const pr.Op,
) void {
    const lhs = ctx.get_value(op.operand(0)).?;
    const rhs = ctx.get_value(op.operand(1)).?;
    const mlir_op = op_fn(ctx.mlir_ctx, lhs, rhs, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_unary(
    comptime op_fn: fn (mlir.Context, mlir.Value, mlir.Location) mlir.Operation,
    ctx: LowerContext,
    op: *const pr.Op,
) void {
    const operand = ctx.get_value(op.operand(0)).?;
    const mlir_op = op_fn(ctx.mlir_ctx, operand, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_convert(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.convert(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_literal(ctx: LowerContext, op: *const pr.Op) void {
    const out_var = op.result(0);
    const out_tensor = out_var.aval.as_tensor();
    const lit = op.params.literal;
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);
    const raw_bytes = switch (lit) {
        inline else => |v| std.mem.asBytes(&v),
    };
    const mlir_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, raw_bytes, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(out_var, mlir_op.result(0));
}

fn lower_reshape(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.reshape(ctx.mlir_ctx, operand, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_transpose(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const perm = op.params.transpose.permutation;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.transpose(ctx.mlir_ctx, operand, out_type, ctx.loc, .{ .permutation = perm });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_broadcast_in_dim(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const bd = op.params.broadcast_in_dim.dimensions;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.broadcast_in_dim(ctx.mlir_ctx, operand, bd, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_iota(ctx: LowerContext, op: *const pr.Op) void {
    const iota_dim = op.params.iota.dimension;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.iota(ctx.mlir_ctx, iota_dim, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_slice(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const sparams = op.params.slice;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.slice(
        ctx.mlir_ctx,
        operand,
        sparams.start_indices,
        sparams.limit_indices,
        sparams.strides,
        out_type,
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_concatenate(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const axis = op.params.concatenate.axis;
    const values = try ctx.arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        values[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;
    }
    const mlir_op = stablehlo.concatenate(ctx.mlir_ctx, values, axis, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_reduce(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const params = op.params.reduce;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const elem_type = dtype_to_dense_elements_type(out_tensor.dtype);

    switch (params.operation) {
        .sum => {
            const zero_bytes = scalar_zero_bytes(out_tensor.dtype);
            const zero_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, zero_bytes, ctx.loc);
            ctx.block.append_operation(zero_op);
            const mlir_op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{zero_op.result(0)}, params.axes, {}, reduce_add_block, ctx.loc);
            ctx.block.append_operation(mlir_op);
            ctx.set_value(op.result(0), mlir_op.result(0));
        },
        .maximum => {
            const min_bytes = scalar_min_bytes(out_tensor.dtype);
            const min_op = stablehlo.constant(ctx.mlir_ctx, &.{}, elem_type, min_bytes, ctx.loc);
            ctx.block.append_operation(min_op);
            const mlir_op = stablehlo.reduce(ctx.mlir_ctx, &.{operand}, &.{min_op.result(0)}, params.axes, {}, reduce_max_block, ctx.loc);
            ctx.block.append_operation(mlir_op);
            ctx.set_value(op.result(0), mlir_op.result(0));
        },
    }
}

fn reduce_add_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.add(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

fn reduce_max_block(_: anytype, ctx: mlir.Context, ins: []const mlir.Value, accs: []const mlir.Value) mlir.Operation {
    return stablehlo.maximum(ctx, ins[0], accs[0], mlir.Location.unknown(ctx));
}

fn lower_dot(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    try lower_fixed_contraction(ctx, op, &.{}, &.{0}, &.{0});
}

fn lower_mm(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    try lower_fixed_contraction(ctx, op, &.{}, &.{1}, &.{0});
}

fn lower_bmm(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const rank = op.operand(0).as_tensor().shape.rank();
    var batch_dims: [pr.max_rank - 2]i64 = undefined;
    for (0..rank - 2) |index| batch_dims[index] = @intCast(index);
    const lhs_contracting: [1]i64 = .{@intCast(rank - 1)};
    const rhs_contracting: [1]i64 = .{@intCast(rank - 2)};
    try lower_fixed_contraction(
        ctx,
        op,
        batch_dims[0 .. rank - 2],
        &lhs_contracting,
        &rhs_contracting,
    );
}

fn lower_fixed_contraction(
    ctx: LowerContext,
    op: *const pr.Op,
    batch_dims: []const i64,
    lhs_contracting: []const i64,
    rhs_contracting: []const i64,
) LowerError!void {
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = batch_dims,
        .rhs_batching_dimensions = batch_dims,
        .lhs_contracting_dimensions = lhs_contracting,
        .rhs_contracting_dimensions = rhs_contracting,
        .precision = .fast,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_dot_general(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const dg = op.params.dot_general;
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const out_tensor = op.result(0).aval.as_tensor();
    const out_type = ctx.tensor_to_mlir_type(out_tensor);
    const mlir_op = stablehlo.dot_general(ctx.mlir_ctx, lhs, rhs, out_type, ctx.loc, .{
        .lhs_batching_dimensions = dg.lhs_batch_dims,
        .rhs_batching_dimensions = dg.rhs_batch_dims,
        .lhs_contracting_dimensions = dg.lhs_contracting_dims,
        .rhs_contracting_dimensions = dg.rhs_contracting_dims,
        .precision = .fast,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_convolution(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const params = op.params.convolution;
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const out_type = ctx.tensor_to_mlir_type(op.result(0).aval.as_tensor());
    const spatial_rank: i64 = @intCast(params.dimensions.input_spatial_dimensions.len);
    const mlir_op = stablehlo.convolution(ctx.mlir_ctx, lhs, rhs, .{
        .window_strides = params.window_strides,
        .pad_value = params.padding,
        .pad_shape = &.{ spatial_rank, 2 },
        .lhs_dilation = params.lhs_dilation,
        .rhs_dilation = params.rhs_dilation,
        .window_reversal = params.window_reversal,
        .input_batch_dimension = params.dimensions.input_batch_dimension,
        .input_feature_dimension = params.dimensions.input_feature_dimension,
        .input_spatial_dimensions = params.dimensions.input_spatial_dimensions,
        .kernel_input_feature_dimension = params.dimensions.kernel_input_feature_dimension,
        .kernel_output_feature_dimension = params.dimensions.kernel_output_feature_dimension,
        .kernel_spatial_dimensions = params.dimensions.kernel_spatial_dimensions,
        .output_batch_dimension = params.dimensions.output_batch_dimension,
        .output_feature_dimension = params.dimensions.output_feature_dimension,
        .output_spatial_dimensions = params.dimensions.output_spatial_dimensions,
        .feature_group_count = params.feature_group_count,
        .batch_group_count = params.batch_group_count,
    }, out_type, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_compare(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const cparams = op.params.compare;
    const lhs = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const rhs = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.compare(
        ctx.mlir_ctx,
        lhs,
        rhs,
        stablehlo.ComparisonDirection.init(ctx.mlir_ctx, map_compare_direction(cparams.direction)),
        stablehlo.CompareType.init(ctx.mlir_ctx, map_compare_type(cparams.compare_type)),
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_select(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const cond = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const on_true = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const on_false = ctx.get_value(op.operand(2)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.select(ctx.mlir_ctx, cond, on_true, on_false, ctx.loc);
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn map_compare_direction(dir: pr.CompareDirection) stablehlo.ComparisonDirection.Direction {
    return switch (dir) {
        .EQ => .EQ,
        .NE => .NE,
        .GE => .GE,
        .GT => .GT,
        .LE => .LE,
        .LT => .LT,
    };
}

fn map_compare_type(ctype: pr.CompareType) stablehlo.CompareType.Type {
    return switch (ctype) {
        .SIGNED => .SIGNED,
        .UNSIGNED => .UNSIGNED,
        .FLOAT => .FLOAT,
        .TOTALORDER => .TOTALORDER,
    };
}

fn lower_gather(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const gparams = op.params.gather;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const indices = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const mlir_op = stablehlo.gather(ctx.mlir_ctx, operand, indices, gparams.slice_sizes, ctx.loc, .{
        .offset_dims = gparams.offset_dims,
        .collapsed_slice_dims = gparams.collapsed_slice_dims,
        .operand_batching_dims = &.{},
        .start_indices_batching_dims = &.{},
        .start_index_map = gparams.start_index_map,
        .index_vector_dim = gparams.index_vector_dim,
    });
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn lower_scatter(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const sparams = op.params.scatter;
    const operand = ctx.get_value(op.operand(0)) orelse return error.InvalidProgram;
    const indices = ctx.get_value(op.operand(1)) orelse return error.InvalidProgram;
    const updates = ctx.get_value(op.operand(2)) orelse return error.InvalidProgram;
    const update_block = try make_update_block(ctx.mlir_ctx, operand.get_type(), ctx.loc, sparams.reduction);
    const mlir_op = stablehlo.scatter(
        ctx.mlir_ctx,
        &.{operand},
        &.{indices},
        &.{updates},
        update_block,
        .{
            .update_window_dims = sparams.update_window_dims,
            .inserted_window_dims = sparams.inserted_window_dims,
            .input_batching_dims = &.{},
            .scatter_indices_batching_dims = &.{},
            .scatter_dims_to_operand_dims = sparams.scatter_dims_to_operand_dims,
            .index_vector_dim = sparams.index_vector_dim,
        },
        ctx.loc,
    );
    ctx.block.append_operation(mlir_op);
    ctx.set_value(op.result(0), mlir_op.result(0));
}

fn make_update_block(ctx: mlir.Context, operand_type: mlir.Type, loc: mlir.Location, reduction: pr.ScatterReduction) error{InvalidMlir}!mlir.Block {
    const elem_type = if (operand_type.as(mlir.RankedTensorType)) |shaped| shaped.get_element_type() else operand_type;
    const arg_type: mlir.Type = .tensor(&.{}, elem_type);
    var block = try mlir.Block.init(&.{ arg_type, arg_type }, &.{ loc, loc });
    const mlir_op = switch (reduction) {
        .add => stablehlo.add(ctx, block.argument(0), block.argument(1), loc),
        .max => stablehlo.maximum(ctx, block.argument(0), block.argument(1), loc),
        .min => stablehlo.minimum(ctx, block.argument(0), block.argument(1), loc),
        .mul => stablehlo.multiply(ctx, block.argument(0), block.argument(1), loc),
    };
    block.append_operation(mlir_op);
    const ret = stablehlo.return_(ctx, mlir_op.result(0), loc);
    block.append_operation(ret);
    return block;
}

fn dtype_to_dense_elements_type(dt: pr.DType) mlir.DenseElementsAttributeTypes {
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i8 => .i8,
        .u8 => .i8,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .i32,
        .u64 => .i64,
        .bool => .bool,
    };
}

fn scalar_zero_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .f16, .bf16 => std.mem.asBytes(&@as(u16, 0)),
        .f32 => std.mem.asBytes(&@as(f32, 0.0)),
        .f64 => std.mem.asBytes(&@as(f64, 0.0)),
        .i8 => std.mem.asBytes(&@as(i8, 0)),
        .u8 => std.mem.asBytes(&@as(u8, 0)),
        .i32 => std.mem.asBytes(&@as(i32, 0)),
        .i64 => std.mem.asBytes(&@as(i64, 0)),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

fn scalar_min_bytes(dtype: pr.DType) []const u8 {
    return switch (dtype) {
        .f16 => std.mem.asBytes(&@as(u16, 0xFC00)), // -inf in f16
        .bf16 => std.mem.asBytes(&pr.DType.bf16.encode(f32, -std.math.inf(f32))),
        .f32 => std.mem.asBytes(&@as(f32, -std.math.inf(f32))),
        .f64 => std.mem.asBytes(&@as(f64, -std.math.inf(f64))),
        .i8 => std.mem.asBytes(&@as(i8, std.math.minInt(i8))),
        .u8 => std.mem.asBytes(&@as(u8, 0)),
        .i32 => std.mem.asBytes(&@as(i32, std.math.minInt(i32))),
        .i64 => std.mem.asBytes(&@as(i64, std.math.minInt(i64))),
        .u32 => std.mem.asBytes(&@as(u32, 0)),
        .u64 => std.mem.asBytes(&@as(u64, 0)),
        .bool => std.mem.asBytes(&@as(bool, false)),
    };
}

/// Configuration for PR to StableHLO lowering.
pub const LowerConfig = struct {
    /// Selects the wire format for StableHLO output.
    ///
    /// Binary is smaller and faster to parse. Text supports direct inspection.
    encoding: stablehlo_types.Encoding = .binary,
};

/// Lowers a PR program into StableHLO bytes allocated by the context allocator.
pub const Lower = struct {
    pub const Input = *pr.Program;
    pub const Output = stablehlo_types.Artifact;

    config: LowerConfig,

    pub fn run(self: Lower, program: Input, ctx: *compilation.Context) !Output {
        const entry = try program.resolve_entry();
        return try lower(
            ctx.allocator,
            program,
            entry,
            self.config.encoding,
        );
    }
};

/// Lower a PR program into a StableHLO artifact.
pub fn lower(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry: pr.FunctionId,
    encoding: stablehlo_types.Encoding,
) !stablehlo_types.Artifact {
    const format: OutputFormat = switch (encoding) {
        .text => .mlir_text,
        .binary => .mlir_bytecode,
    };

    const bytes = try lower_program_to_mlir(allocator, program, entry, format);

    return .{ .bytes = bytes, .encoding = encoding };
}

// Tests.

const kernel_test = @import("kernel");

fn outline_kernel_requests_for_test(program: *pr.Program) !kernel_test.ExtractedCandidates {
    const Provider = struct {
        fn compile(
            _: *anyopaque,
            _: pr.Function,
            _: @import("device").Device,
            _: std.mem.Allocator,
        ) kernel_test.CompileError!kernel_test.Artifact {
            return error.Unsupported;
        }
    };
    var provider_context: u8 = 0;
    const providers = [_]kernel_test.KernelProvider{.{
        .name = "mock",
        .compiler = .{
            .context = @ptrCast(&provider_context),
            .vtable = &.{ .compile = Provider.compile },
        },
    }};
    var discovered = kernel_test.Candidates.init(std.testing.allocator);
    defer discovered.deinit();
    var extracted = kernel_test.ExtractedCandidates.init(std.testing.allocator);
    errdefer extracted.deinit();
    var ctx = compilation.Context{ .allocator = std.testing.allocator, .io = std.testing.io };
    _ = try (kernelize.DiscoverCandidates{
        .providers = &providers,
        .candidates = &discovered,
    }).run(program, &ctx);
    _ = try (kernelize.ExtractCandidates{
        .candidates = &discovered,
        .extracted = &extracted,
    }).run(program, &ctx);
    return extracted;
}

test "lowering produces verified bytecode" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    const entry_id = blk: {
        var b = try pr.FunctionBuilder.init(&program, "main");
        defer b.deinit();
        const a = try b.param_tensor(.f32, &.{ 2, 3 });
        const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
        const c = try b.param_tensor(.f32, &.{ 2, 2 });
        const mm_id = try b.mm(a, b_id);
        const add_id = try b.add(mm_id, c);
        const out_id = try b.multiply(add_id, c);
        const func = try b.finish(.{ .returns = &.{out_id} });
        break :blk try program.add_function(func);
    };

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, entry_id, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports fixed contraction forms" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    {
        var b = try pr.FunctionBuilder.init(&program, "dot");
        defer b.deinit();
        const lhs = try b.param_tensor(.f32, &.{3});
        const rhs = try b.param_tensor(.f32, &.{3});
        const out = try b.dot(lhs, rhs);
        const bytes = try lower_function_to_mlir(
            std.testing.allocator,
            try b.finish(.{ .returns = &.{out} }),
            .mlir_bytecode,
        );
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(bytes.len > 0);
    }
    {
        var b = try pr.FunctionBuilder.init(&program, "mm");
        defer b.deinit();
        const lhs = try b.param_tensor(.f32, &.{ 2, 3 });
        const rhs = try b.param_tensor(.f32, &.{ 3, 4 });
        const out = try b.mm(lhs, rhs);
        const bytes = try lower_function_to_mlir(
            std.testing.allocator,
            try b.finish(.{ .returns = &.{out} }),
            .mlir_bytecode,
        );
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(bytes.len > 0);
    }
    {
        var b = try pr.FunctionBuilder.init(&program, "bmm");
        defer b.deinit();
        const lhs = try b.param_tensor(.f32, &.{ 2, 5, 3, 4 });
        const rhs = try b.param_tensor(.f32, &.{ 2, 5, 4, 6 });
        const out = try b.bmm(lhs, rhs);
        const bytes = try lower_function_to_mlir(
            std.testing.allocator,
            try b.finish(.{ .returns = &.{out} }),
            .mlir_bytecode,
        );
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(bytes.len > 0);
    }
}

test "lowering keeps non-entry functions" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var fwd_builder = try pr.FunctionBuilder.init(&program, "forward");
    defer fwd_builder.deinit();
    const fwd_x = try fwd_builder.param_tensor(.f32, &.{2});
    const fwd = try fwd_builder.finish(.{ .returns = &.{fwd_x} });
    _ = try program.add_function(fwd);

    var bwd_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer bwd_builder.deinit();
    const bwd_x = try bwd_builder.param_tensor(.f32, &.{2});
    const bwd = try bwd_builder.finish(.{ .returns = &.{bwd_x} });
    const bwd_id = try program.add_function(bwd);

    const text = try lower_program_to_mlir(testing.allocator, &program, bwd_id, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @forward") != null);
}

test "lowering renames non-entry main" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    const checkpoint = program.checkpoint_appends();
    var removed_builder = try pr.FunctionBuilder.init(&program, "removed");
    defer removed_builder.deinit();
    _ = try program.add_function(try removed_builder.finish(.{ .returns = &.{} }));
    program.restore_appends(checkpoint);

    var main_builder = try pr.FunctionBuilder.init(&program, "main");
    defer main_builder.deinit();
    const main_x = try main_builder.param_tensor(.f32, &.{2});
    const main_fn = try main_builder.finish(.{ .returns = &.{main_x} });
    const main_id = try program.add_function(main_fn);

    var other_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer other_builder.deinit();
    const other_x = try other_builder.param_tensor(.f32, &.{2});
    const other_call = try other_builder.call(main_id, &.{other_x});
    const other_fn = try other_builder.finish(.{ .returns = other_call.outputs });
    const other_id = try program.add_function(other_fn);

    const text = try lower_program_to_mlir(testing.allocator, &program, other_id, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @main_non_entry_0") != null);
    try testing.expect(std.mem.indexOf(u8, text, "call @main_non_entry_0") != null);
}

test "lowering supports reshape/broadcast/transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const y = try b.broadcast_in_dim(r, &.{ 2, 6 }, &.{1});

    const func = try b.finish(.{ .returns = &.{y} });
    const entry_id = try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, entry_id, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports custom_call" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const outputs = (try b.custom_call(.{
        .target_name = "zigrad.test.missing_handler",
        .has_side_effect = false,
    }, &.{x}, &.{x.aval})).outputs;
    const y = outputs[0];

    const func = try b.finish(.{ .returns = &.{y} });
    const entry_id = try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, entry_id, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports multi-output custom_call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("mock_multi", &.{kernel_test.provider_annotation("mock")});
    const ex = try b.emit(.{ .exp = {} }, &.{x});
    const lg = try b.emit(.{ .log = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{ ex, lg } });
    const entry_id = try program.add_function(func);
    var candidates = try outline_kernel_requests_for_test(&program);
    defer candidates.deinit();

    const selected_device = @import("device").Device{ .platform = .cpu };
    const candidate = candidates.entries.items[0];
    const outlined = program.get_function_by_id(candidate.callable_function).?;
    const function_fingerprint = try fingerprint.function(testing.allocator, outlined);
    const selection_key = try kernel_test.make_selection_key(
        testing.allocator,
        candidate.boundary,
        selected_device,
        function_fingerprint,
    );
    defer testing.allocator.free(selection_key.bytes);

    var store = kernel_test.KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put(selection_key, .{
        .candidate = .{ .provider = .{
            .provider_name = "mock",
            .artifact = .{ .data = try testing.allocator.dupe(u8, "mock") },
        } },
        .reason = "available",
    });

    var kp = kernelize.KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = selected_device,
    };

    var pass_ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &pass_ctx);

    const text = try lower_program_to_mlir(testing.allocator, &program, entry_id, .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "stablehlo.custom_call") != null);
    try testing.expect(std.mem.indexOf(u8, text, stablehlo_types.custom_call_payload_name) != null);
    try testing.expect(std.mem.indexOf(u8, text, "tensor<2xf32>, tensor<2xf32>") != null);
}

test "lowering supports vjp matmul demo" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();
    const fwd_id = blk: {
        var b = try pr.FunctionBuilder.init(&program, "main");
        defer b.deinit();
        const a = try b.param_tensor(.f32, &.{ 2, 3 });
        const b_id = try b.param_tensor(.f32, &.{ 3, 2 });
        const c = try b.param_tensor(.f32, &.{ 2, 2 });
        const mm_id = try b.mm(a, b_id);
        const add_id = try b.add(mm_id, c);
        const out_id = try b.multiply(add_id, c);
        const func = try b.finish(.{ .returns = &.{out_id} });
        break :blk try program.add_function(func);
    };

    const vjp_id = try @import("pr").ad.vjp(std.testing.allocator, &program, fwd_id, "vjp", .{});
    const bc = try lower_program_to_mlir(std.testing.allocator, &program, vjp_id, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering can outline via region annotation" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    try b.push_region("outlined-mm", &.{outline.annotation});
    const d = try b.mm(a, c);
    try b.pop_region();
    const func = try b.finish(.{ .returns = &.{d} });
    const entry_id = try program.add_function(func);

    var pass_ctx = compilation.Context{ .allocator = std.testing.allocator, .io = std.testing.io };
    _ = try (outline.Pass{}).run(&program, &pass_ctx);

    const text = try lower_program_to_mlir(std.testing.allocator, &program, entry_id, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "call @main_outlined_0") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "llvm.noinline") == null);
}

test "lower convolution" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();
    const image = try b.param_tensor(.f32, &.{ 1, 8, 8, 3 });
    const kernel = try b.param_tensor(.f32, &.{ 3, 3, 3, 8 });
    const output = try b.convolution(image, kernel, .{
        .window_strides = &.{ 2, 2 },
        .padding = &.{ 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1 },
        .rhs_dilation = &.{ 1, 1 },
        .window_reversal = &.{ false, false },
        .dimensions = .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 3,
            .input_spatial_dimensions = &.{ 1, 2 },
            .kernel_input_feature_dimension = 2,
            .kernel_output_feature_dimension = 3,
            .kernel_spatial_dimensions = &.{ 0, 1 },
            .output_batch_dimension = 0,
            .output_feature_dimension = 3,
            .output_spatial_dimensions = &.{ 1, 2 },
        },
    });
    const func = try b.finish(.{ .returns = &.{output} });
    const entry_id = try program.add_function(func);

    const text = try lower_program_to_mlir(testing.allocator, &program, entry_id, .mlir_text);
    defer testing.allocator.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "stablehlo.convolution") != null);
    try testing.expect(std.mem.indexOf(u8, text, "dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]") != null);

    const vjp_id = try @import("pr").ad.vjp(testing.allocator, &program, entry_id, "vjp", .{});
    const vjp = try lower_program_to_mlir(testing.allocator, &program, vjp_id, .mlir_bytecode);
    defer testing.allocator.free(vjp);
    if (vjp.len == 0) return error.EmptyConvolutionVjp;
}

test "lower operation produces StableHLO" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 3, 2 });
    const z = try b.mm(x, y);
    const func = try b.finish(.{ .returns = &.{z} });

    const entry_id = try program.add_function(func);
    var output = try lower(std.testing.allocator, &program, entry_id, .binary);
    defer output.deinit(std.testing.allocator);

    try std.testing.expect(output.bytes.len > 0);
}

test "lower operation emits stablehlo.custom_call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const outputs = (try b.custom_call(.{
        .target_name = "zigrad.test.missing_handler",
        .has_side_effect = false,
    }, &.{x}, &.{x.aval})).outputs;
    const y = outputs[0];
    const func = try b.finish(.{ .returns = &.{y} });
    const entry_id = try program.add_function(func);

    var artifact = try lower(testing.allocator, &program, entry_id, .text);
    defer artifact.deinit(testing.allocator);

    try testing.expect(std.mem.indexOf(u8, artifact.bytes, "stablehlo.custom_call") != null);
}
