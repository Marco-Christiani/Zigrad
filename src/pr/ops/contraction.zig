//! Contraction Operations
//! Ops that contract dimensions (matrix multiply, convolution).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;

const log = std.log.scoped(.@"zg/contraction");

// Vector dot product

pub const dot = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
        if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;

        const lhs = op.operand(0).as_tensor();
        const rhs = op.operand(1).as_tensor();
        const out = op.result(0).as_tensor();

        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotTypeMismatch;
        if (lhs.shape.rank() != 1 or rhs.shape.rank() != 1 or out.shape.rank() != 0)
            return error.DotTypeMismatch;
        if (lhs.shape.dims[0] != rhs.shape.dims[0]) return error.DotTypeMismatch;
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
        if (inputs.len != 2) return error.InvalidOpArity;

        const lhs = inputs[0].as_tensor();
        const rhs = inputs[1].as_tensor();

        if (lhs.dtype != rhs.dtype) return error.DotTypeMismatch;
        if (lhs.shape.rank() != 1 or rhs.shape.rank() != 1) return error.DotTypeMismatch;
        if (lhs.shape.dims[0] != rhs.shape.dims[0]) return error.DotTypeMismatch;
        return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{
            .dims = try alloc.alloc(i64, 0),
        } } };
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const lhs_primal = try ctx.primal(op.operand(0));
        const rhs_primal = try ctx.primal(op.operand(1));

        const shape = lhs_primal.as_tensor().shape.dims;
        const expanded = try ctx.builder.broadcast_in_dim(out_cot, shape, &.{});
        const lhs_contrib = try ctx.builder.multiply(expanded, rhs_primal);
        const rhs_contrib = try ctx.builder.multiply(expanded, lhs_primal);

        try ctx.add_cotangent(op.operand(0), lhs_contrib);
        try ctx.add_cotangent(op.operand(1), rhs_contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{dot}(a, b)) =
    ///  \operatorname{dot}(\mathrm{d}a, b) + \operatorname{dot}(a, \mathrm{d}b)\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const a = try ctx.primal(op.operand(0));
        const b = try ctx.primal(op.operand(1));
        if (ctx.tangent(op.operand(0))) |da| {
            try ctx.add_tangent(op.result(0), try ctx.builder.dot(da, b));
        }
        if (ctx.tangent(op.operand(1))) |db| {
            try ctx.add_tangent(op.result(0), try ctx.builder.dot(a, db));
        }
    }

    pub fn format(writer: *types.Writer, op: *const pr.Op, _: void) types.FormatError!void {
        if (op.inputs.len == 0) return;
        const lhs = op.operand(0).as_tensor();
        try writer.print("K={d}", .{lhs.shape.dims[0]});
    }
};

/// Rank-two matrix multiplication.
pub const mm = matrix_multiply(false);

/// Prefix-batched matrix multiplication with one or more batch dimensions.
pub const bmm = matrix_multiply(true);

fn matrix_multiply(comptime batched: bool) type {
    return struct {
        pub const arity = .{ .in = 2, .out = 1 };

        fn type_mismatch() pr.ValidationError {
            return if (batched) error.BMMTypeMismatch else error.MMTypeMismatch;
        }

        pub fn validate(op: *const pr.Op, _: void) pr.ValidationError!void {
            if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;
            const lhs = op.operand(0).as_tensor();
            const rhs = op.operand(1).as_tensor();
            const out = op.result(0).as_tensor();
            const expected_rank: usize = if (batched) 3 else 2;
            if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype)
                return type_mismatch();
            if (lhs.shape.rank() != rhs.shape.rank() or
                lhs.shape.rank() != out.shape.rank() or
                lhs.shape.rank() < expected_rank or
                (!batched and lhs.shape.rank() != expected_rank))
                return type_mismatch();
            const rank = lhs.shape.rank();
            for (0..rank - 2) |index| {
                if (lhs.shape.dims[index] != rhs.shape.dims[index] or
                    lhs.shape.dims[index] != out.shape.dims[index])
                    return type_mismatch();
            }
            if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2] or
                out.shape.dims[rank - 2] != lhs.shape.dims[rank - 2] or
                out.shape.dims[rank - 1] != rhs.shape.dims[rank - 1])
                return type_mismatch();
        }

        pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, _: void) pr.BuildError!Aval {
            if (inputs.len != 2) return error.InvalidOpArity;
            const lhs = inputs[0].as_tensor();
            const rhs = inputs[1].as_tensor();
            const expected_rank: usize = if (batched) 3 else 2;
            if (lhs.dtype != rhs.dtype or lhs.shape.rank() != rhs.shape.rank() or
                lhs.shape.rank() < expected_rank or
                (!batched and lhs.shape.rank() != expected_rank))
                return if (batched) error.BMMTypeMismatch else error.MMTypeMismatch;
            const rank = lhs.shape.rank();
            for (0..rank - 2) |index| {
                if (lhs.shape.dims[index] != rhs.shape.dims[index])
                    return if (batched) error.BMMTypeMismatch else error.MMTypeMismatch;
            }
            if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2])
                return if (batched) error.BMMTypeMismatch else error.MMTypeMismatch;
            const dims = try alloc.dupe(i64, lhs.shape.dims);
            dims[rank - 1] = rhs.shape.dims[rank - 1];
            return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{ .dims = dims } } };
        }

        pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: void) types.AdError!void {
            const out_cot = ctx.cotangent(op.result(0)) orelse return;
            const lhs = try ctx.primal(op.operand(0));
            const rhs = try ctx.primal(op.operand(1));
            var permutation: [pr.max_rank]i64 = undefined;
            const rank = lhs.as_tensor().shape.rank();
            for (0..rank) |index| permutation[index] = @intCast(index);
            std.mem.swap(i64, &permutation[rank - 2], &permutation[rank - 1]);
            const lhs_t = try ctx.builder.transpose(lhs, permutation[0..rank]);
            const rhs_t = try ctx.builder.transpose(rhs, permutation[0..rank]);
            const lhs_contrib = if (batched)
                try ctx.builder.bmm(out_cot, rhs_t)
            else
                try ctx.builder.mm(out_cot, rhs_t);
            const rhs_contrib = if (batched)
                try ctx.builder.bmm(lhs_t, out_cot)
            else
                try ctx.builder.mm(lhs_t, out_cot);
            try ctx.add_cotangent(op.operand(0), lhs_contrib);
            try ctx.add_cotangent(op.operand(1), rhs_contrib);
        }

        pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: void) types.AdError!void {
            const lhs = try ctx.primal(op.operand(0));
            const rhs = try ctx.primal(op.operand(1));
            if (ctx.tangent(op.operand(0))) |lhs_tangent| {
                const term = if (batched)
                    try ctx.builder.bmm(lhs_tangent, rhs)
                else
                    try ctx.builder.mm(lhs_tangent, rhs);
                try ctx.add_tangent(op.result(0), term);
            }
            if (ctx.tangent(op.operand(1))) |rhs_tangent| {
                const term = if (batched)
                    try ctx.builder.bmm(lhs, rhs_tangent)
                else
                    try ctx.builder.mm(lhs, rhs_tangent);
                try ctx.add_tangent(op.result(0), term);
            }
        }

        pub fn format(writer: *types.Writer, op: *const pr.Op, _: void) types.FormatError!void {
            if (op.inputs.len == 0) return;
            const shape = op.operand(0).as_tensor().shape.dims;
            try writer.print("M={d}, K={d}, N={d}", .{
                shape[shape.len - 2],
                shape[shape.len - 1],
                op.operand(1).as_tensor().shape.dims[shape.len - 1],
            });
        }
    };
}

// Dot General

pub const convolution = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, params: pr.ConvolutionParams) pr.ValidationError!void {
        if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;
        const lhs = op.operand(0).as_tensor();
        const rhs = op.operand(1).as_tensor();
        const out = op.result(0).as_tensor();
        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.ConvolutionTypeMismatch;

        var dims: [max_rank]i64 = undefined;
        const expected = compute_convolution_output_dims(lhs, rhs, params, &dims) orelse
            return error.ConvolutionTypeMismatch;
        if (!std.mem.eql(i64, expected, out.shape.dims)) return error.ConvolutionTypeMismatch;
    }

    pub fn infer_output(
        allocator: std.mem.Allocator,
        inputs: []const *pr.Var,
        params: pr.ConvolutionParams,
    ) pr.BuildError!Aval {
        if (inputs.len != 2) return error.InvalidOpArity;
        const lhs = inputs[0].as_tensor();
        const rhs = inputs[1].as_tensor();
        if (lhs.dtype != rhs.dtype) return error.ConvolutionTypeMismatch;

        var dims: [max_rank]i64 = undefined;
        const computed = compute_convolution_output_dims(lhs, rhs, params, &dims) orelse
            return error.ConvolutionTypeMismatch;
        return .{ .tensor = .{
            .dtype = lhs.dtype,
            .shape = .{ .dims = try allocator.dupe(i64, computed) },
        } };
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, params: pr.ConvolutionParams) types.AdError!void {
        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        if (params.feature_group_count != 1 or params.batch_group_count != 1) return error.UnsupportedDerivative;
        for (params.window_reversal) |reversed| if (reversed) return error.UnsupportedDerivative;

        const lhs = try ctx.primal(op.operand(0));
        const rhs = try ctx.primal(op.operand(1));
        const lhs_shape = op.operand(0).as_tensor().shape.dims;
        const rhs_shape = op.operand(1).as_tensor().shape.dims;
        const out_shape = op.result(0).as_tensor().shape.dims;
        const spatial_rank = params.dimensions.input_spatial_dimensions.len;

        var lhs_padding: [max_rank * 2]i64 = undefined;
        var rhs_padding: [max_rank * 2]i64 = undefined;
        var reverse_window: [max_rank]bool = undefined;
        @memset(reverse_window[0..spatial_rank], true);
        for (0..spatial_rank) |i| {
            const input_size = lhs_shape[@intCast(params.dimensions.input_spatial_dimensions[i])];
            const kernel_size = rhs_shape[@intCast(params.dimensions.kernel_spatial_dimensions[i])];
            const output_size = out_shape[@intCast(params.dimensions.output_spatial_dimensions[i])];
            const dilated_input = dilated_size(input_size, params.lhs_dilation[i]);
            const dilated_kernel = dilated_size(kernel_size, params.rhs_dilation[i]);
            const dilated_output = dilated_size(output_size, params.window_strides[i]);
            const low = params.padding[i * 2];

            lhs_padding[i * 2] = dilated_kernel - low - 1;
            lhs_padding[i * 2 + 1] = dilated_input + dilated_kernel - 1 -
                dilated_output - lhs_padding[i * 2];
            rhs_padding[i * 2] = low;
            rhs_padding[i * 2 + 1] = dilated_output - dilated_input + dilated_kernel - low - 1;
        }

        const lhs_cot = try ctx.builder.convolution(out_cot, rhs, .{
            .window_strides = params.lhs_dilation,
            .padding = lhs_padding[0 .. spatial_rank * 2],
            .lhs_dilation = params.window_strides,
            .rhs_dilation = params.rhs_dilation,
            .window_reversal = reverse_window[0..spatial_rank],
            .dimensions = .{
                .input_batch_dimension = params.dimensions.output_batch_dimension,
                .input_feature_dimension = params.dimensions.output_feature_dimension,
                .input_spatial_dimensions = params.dimensions.output_spatial_dimensions,
                .kernel_input_feature_dimension = params.dimensions.kernel_output_feature_dimension,
                .kernel_output_feature_dimension = params.dimensions.kernel_input_feature_dimension,
                .kernel_spatial_dimensions = params.dimensions.kernel_spatial_dimensions,
                .output_batch_dimension = params.dimensions.input_batch_dimension,
                .output_feature_dimension = params.dimensions.input_feature_dimension,
                .output_spatial_dimensions = params.dimensions.input_spatial_dimensions,
            },
        });
        const rhs_cot = try ctx.builder.convolution(lhs, out_cot, .{
            .window_strides = params.rhs_dilation,
            .padding = rhs_padding[0 .. spatial_rank * 2],
            .lhs_dilation = params.lhs_dilation,
            .rhs_dilation = params.window_strides,
            .window_reversal = params.window_reversal,
            .dimensions = .{
                .input_batch_dimension = params.dimensions.input_feature_dimension,
                .input_feature_dimension = params.dimensions.input_batch_dimension,
                .input_spatial_dimensions = params.dimensions.input_spatial_dimensions,
                .kernel_input_feature_dimension = params.dimensions.output_batch_dimension,
                .kernel_output_feature_dimension = params.dimensions.output_feature_dimension,
                .kernel_spatial_dimensions = params.dimensions.output_spatial_dimensions,
                .output_batch_dimension = params.dimensions.kernel_input_feature_dimension,
                .output_feature_dimension = params.dimensions.kernel_output_feature_dimension,
                .output_spatial_dimensions = params.dimensions.kernel_spatial_dimensions,
            },
        });
        try ctx.add_cotangent(op.operand(0), lhs_cot);
        try ctx.add_cotangent(op.operand(1), rhs_cot);
    }

    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, params: pr.ConvolutionParams) types.AdError!void {
        const lhs = try ctx.primal(op.operand(0));
        const rhs = try ctx.primal(op.operand(1));
        if (ctx.tangent(op.operand(0))) |lhs_tangent| {
            try ctx.add_tangent(
                op.result(0),
                try ctx.builder.convolution(lhs_tangent, rhs, params),
            );
        }
        if (ctx.tangent(op.operand(1))) |rhs_tangent| {
            try ctx.add_tangent(
                op.result(0),
                try ctx.builder.convolution(lhs, rhs_tangent, params),
            );
        }
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, params: pr.ConvolutionParams) types.FormatError!void {
        try writer.print("strides={any}, padding={any}, feature_groups={d}, batch_groups={d}", .{
            params.window_strides,
            params.padding,
            params.feature_group_count,
            params.batch_group_count,
        });
    }
};

pub const dot_general = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, dg_params: pr.DotGeneralParams) pr.ValidationError!void {
        if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;

        const lhs = op.operand(0).as_tensor();
        const rhs = op.operand(1).as_tensor();
        const out = op.result(0).as_tensor();
        if (lhs.dtype != rhs.dtype or lhs.dtype != out.dtype) return error.DotGeneralTypeMismatch;

        var dims_buf: [max_rank]i64 = undefined;
        const expected = compute_dot_general_output_dims(lhs, rhs, dg_params, &dims_buf) orelse {
            log.err(
                "dot_general shape mismatch: lhs={any} rhs={any} out={any} batch(lhs={any}, rhs={any}) contract(lhs={any}, rhs={any})",
                .{
                    lhs.shape.dims,
                    rhs.shape.dims,
                    out.shape.dims,
                    dg_params.lhs_batch_dims,
                    dg_params.rhs_batch_dims,
                    dg_params.lhs_contracting_dims,
                    dg_params.rhs_contracting_dims,
                },
            );
            return error.DotGeneralTypeMismatch;
        };
        if (!std.mem.eql(i64, out.shape.dims, expected)) {
            log.err(
                "dot_general out dims mismatch: expected={any} actual={any}",
                .{ expected, out.shape.dims },
            );
            return error.DotGeneralTypeMismatch;
        }
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, dg_params: pr.DotGeneralParams) pr.BuildError!Aval {
        if (inputs.len != 2) return error.InvalidOpArity;
        const lhs = inputs[0].as_tensor();
        const rhs = inputs[1].as_tensor();
        if (lhs.dtype != rhs.dtype) return error.DotGeneralTypeMismatch;
        var dims_buf: [max_rank]i64 = undefined;
        const computed = compute_dot_general_output_dims(lhs, rhs, dg_params, &dims_buf) orelse
            return error.DotGeneralTypeMismatch;
        const out_dims = try alloc.dupe(i64, computed);
        return .{ .tensor = .{ .dtype = lhs.dtype, .shape = .{ .dims = out_dims } } };
    }

    /// VJP backward for dot_general.
    ///
    /// Tries three strategies in order:
    ///  1. `maybe_batched_matmul_vjp` - fast path for single-contracting-dim batched matmul.
    ///  2. `maybe_general_dot_vjp` - general case with arbitrary batch/contracting dims.
    ///  3. Inline 3-rank special case: lhs is [B,M,K], rhs is [K,N] or [N,K],
    ///      no batch dims, single contracting dim at lhs position 2. Flattens
    ///      the batch and row dims to reduce to an `mm`, then reshapes back.
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, dg_params: pr.DotGeneralParams) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const lhs_primal = try ctx.primal(op.operand(0));
        const rhs_primal = try ctx.primal(op.operand(1));

        const lhs_contrib, const rhs_contrib = blk: {
            if (try maybe_batched_matmul_vjp(ctx, op.operand(0), op.operand(1), out_cot, lhs_primal, rhs_primal, dg_params)) |pair| {
                break :blk .{ pair.lhs, pair.rhs };
            }

            if (try maybe_general_dot_vjp(ctx, op.operand(0), op.operand(1), out_cot, lhs_primal, rhs_primal, dg_params)) |pair| {
                break :blk .{ pair.lhs, pair.rhs };
            }

            // 3-rank special case: lhs=[B,M,K] @ rhs=[K,N] (or [N,K]) => out=[B,M,N].
            // No batch dims, single contracting dim at lhs position 2.
            // Strategy: flatten B*M into a single dim, do 2D matmul, reshape back.
            if (dg_params.lhs_batch_dims.len == 0 and dg_params.rhs_batch_dims.len == 0 and
                dg_params.lhs_contracting_dims.len == 1 and dg_params.lhs_contracting_dims[0] == 2 and
                dg_params.rhs_contracting_dims.len == 1 and
                (dg_params.rhs_contracting_dims[0] == 0 or dg_params.rhs_contracting_dims[0] == 1))
            {
                const lhs_t = op.operand(0).as_tensor();
                const rhs_t = op.operand(1).as_tensor();
                const out_t = op.result(0).as_tensor();

                if (lhs_t.shape.rank() != 3 or rhs_t.shape.rank() != 2 or out_t.shape.rank() != 3) {
                    return error.UnsupportedDerivative;
                }

                const b = out_t.shape.dims[0];
                const m = out_t.shape.dims[1];
                const k = lhs_t.shape.dims[2];
                const rhs_contract_dim = dg_params.rhs_contracting_dims[0];
                const n = if (rhs_contract_dim == 0) rhs_t.shape.dims[1] else rhs_t.shape.dims[0];
                const bm = b * m;

                // Flatten out_cot [B,M,N] to [B*M, N] for 2D matmul.
                const out2 = try ctx.builder.reshape(out_cot, &.{ bm, n });

                // d_lhs = out2 @ rhs^T => [B*M, K] => reshape back to [B,M,K].
                // If rhs is [K,N] (contract_dim=0), transpose to [N,K] first.
                // If rhs is [N,K] (contract_dim=1), it's already in the right layout.
                const lhs_c = if (rhs_contract_dim == 0) lhs_blk: {
                    const rhs_t2 = try ctx.builder.transpose(rhs_primal, &.{ 1, 0 });
                    const lhs_flat = try ctx.builder.mm(out2, rhs_t2);
                    break :lhs_blk try ctx.builder.reshape(lhs_flat, &.{ b, m, k });
                } else lhs_blk: {
                    const lhs_flat = try ctx.builder.mm(out2, rhs_primal);
                    break :lhs_blk try ctx.builder.reshape(lhs_flat, &.{ b, m, k });
                };

                // d_rhs = lhs^T @ out2: flatten lhs [B,M,K] to [B*M, K], transpose to [K, B*M],
                // then matmul with out2 [B*M, N] => [K, N].
                const lhs2 = try ctx.builder.reshape(lhs_primal, &.{ bm, k });
                const lhs2_t = try ctx.builder.transpose(lhs2, &.{ 1, 0 });
                // If rhs was [K,N] (contract_dim=0), result [K,N] is already correct.
                // If rhs was [N,K] (contract_dim=1), transpose [K,N] -> [N,K] to match.
                const rhs_c = if (rhs_contract_dim == 0) rhs_blk: {
                    break :rhs_blk try ctx.builder.mm(lhs2_t, out2);
                } else rhs_blk: {
                    const rhs_k_n = try ctx.builder.mm(lhs2_t, out2);
                    break :rhs_blk try ctx.builder.transpose(rhs_k_n, &.{ 1, 0 });
                };

                break :blk .{ lhs_c, rhs_c };
            }

            return error.UnsupportedDerivative;
        };

        try ctx.add_cotangent(op.operand(0), lhs_contrib);
        try ctx.add_cotangent(op.operand(1), rhs_contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{dot\_general}(A, B, p)) =
    ///  \operatorname{dot\_general}(\mathrm{d}A, B, p) +
    ///  \operatorname{dot\_general}(A, \mathrm{d}B, p)\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, dg_params: pr.DotGeneralParams) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const a = try ctx.primal(op.operand(0));
        const b = try ctx.primal(op.operand(1));
        if (ctx.tangent(op.operand(0))) |da| {
            try ctx.add_tangent(op.result(0), try ctx.builder.dot_general(da, b, dg_params));
        }
        if (ctx.tangent(op.operand(1))) |db| {
            try ctx.add_tangent(op.result(0), try ctx.builder.dot_general(a, db, dg_params));
        }
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, dg_params: pr.DotGeneralParams) types.FormatError!void {
        try writer.writeAll("batch=(lhs=");
        try write_dims(writer, dg_params.lhs_batch_dims);
        try writer.writeAll(", rhs=");
        try write_dims(writer, dg_params.rhs_batch_dims);
        try writer.writeAll(") contracting=(lhs=");
        try write_dims(writer, dg_params.lhs_contracting_dims);
        try writer.writeAll(", rhs=");
        try write_dims(writer, dg_params.rhs_contracting_dims);
        try writer.writeAll(")");
    }
};

const BatchedMatmulVjpPair = struct { lhs: *pr.Var, rhs: *pr.Var };

/// General-case VJP for dot_general with arbitrary batch/contracting dims.
///
/// Classifies each dimension as batch, contracting, or "other" (free), then
///  computes cotangent contributions for lhs and rhs by contracting the output
///  cotangent against the opposite primal. The canonical outputs are in
///  [batch..., other..., contracting...] order, so `transpose_to_match_multi`
///  permutes them back to the original operand layouts.
///
/// Returns null if the dimension configuration is unsupported (e.g. mismatched
///  batch/contracting counts or zero contracting dims), letting the caller fall
///  through to more specialized strategies.
fn maybe_general_dot_vjp(
    ctx: types.VjpContext,
    lhs_var: *const pr.Var,
    rhs_var: *const pr.Var,
    out_cot: *pr.Var,
    lhs_primal: *pr.Var,
    rhs_primal: *pr.Var,
    params: pr.DotGeneralParams,
) types.AdError!?BatchedMatmulVjpPair {
    if (params.lhs_contracting_dims.len != params.rhs_contracting_dims.len) return null;
    if (params.lhs_contracting_dims.len == 0) return null;
    if (params.lhs_batch_dims.len != params.rhs_batch_dims.len) return null;

    const lhs_t = lhs_var.as_tensor();
    const rhs_t = rhs_var.as_tensor();
    const out_t = out_cot.as_tensor();

    const lhs_rank = lhs_t.shape.rank();
    const rhs_rank = rhs_t.shape.rank();
    const batch_len: usize = params.lhs_batch_dims.len;

    // Classify dims: each dimension of lhs/rhs is batch, contracting, or "other" (free).
    // Output layout is [batch..., lhs_other..., rhs_other...].
    const lhs_other = try collect_other_dims(ctx.allocator, lhs_rank, params.lhs_batch_dims, params.lhs_contracting_dims);
    defer ctx.allocator.free(lhs_other);
    const rhs_other = try collect_other_dims(ctx.allocator, rhs_rank, params.rhs_batch_dims, params.rhs_contracting_dims);
    defer ctx.allocator.free(rhs_other);

    const expected_out_rank = batch_len + lhs_other.len + rhs_other.len;
    if (out_t.shape.rank() != expected_out_rank) return error.UnsupportedDerivative;

    // Output dim ranges: [0..batch_len) are batch, [batch_len..+lhs_other) are lhs free,
    // [batch_len+lhs_other..end) are rhs free.
    const out_batch = try build_range(ctx.allocator, 0, batch_len);
    defer ctx.allocator.free(out_batch);

    // d_lhs: contract out_cot with rhs_primal. We contract over the rhs "other" dims
    // (which appear at positions [batch_len+lhs_other.len..] in the output).
    const rhs_contract_from_out = try build_range(ctx.allocator, batch_len + lhs_other.len, rhs_other.len);
    defer ctx.allocator.free(rhs_contract_from_out);

    const d_lhs_canon = try ctx.builder.dot_general(out_cot, rhs_primal, .{
        .lhs_batch_dims = out_batch,
        .rhs_batch_dims = params.rhs_batch_dims,
        .lhs_contracting_dims = rhs_contract_from_out,
        .rhs_contracting_dims = rhs_other,
    });

    // Canonical result is [batch..., lhs_other..., lhs_contracting...], permute
    //  back to the original lhs dim order.
    const d_lhs = try transpose_to_match_multi(
        ctx,
        d_lhs_canon,
        lhs_rank,
        params.lhs_batch_dims,
        lhs_other,
        params.lhs_contracting_dims,
    );

    // d_rhs: contract lhs_primal with out_cot. We contract over the lhs "other" dims
    // (which appear at positions [batch_len..batch_len+lhs_other.len) in the output).
    const lhs_contract = lhs_other;
    const out_contract = try build_range(ctx.allocator, batch_len, lhs_other.len);
    defer ctx.allocator.free(out_contract);

    const d_rhs_canon = try ctx.builder.dot_general(lhs_primal, out_cot, .{
        .lhs_batch_dims = params.lhs_batch_dims,
        .rhs_batch_dims = out_batch,
        .lhs_contracting_dims = lhs_contract,
        .rhs_contracting_dims = out_contract,
    });

    // Canonical result is [batch..., rhs_contracting..., rhs_other...], permute
    // back to the original rhs dim order.
    const d_rhs = try transpose_to_match_multi(
        ctx,
        d_rhs_canon,
        rhs_rank,
        params.rhs_batch_dims,
        params.rhs_contracting_dims,
        rhs_other,
    );

    return .{ .lhs = d_lhs, .rhs = d_rhs };
}

/// Fast-path VJP for batched matmul: exactly one contracting dim per operand
///  and at least one batch dim. Each operand has exactly one "other" (free)
///  dimension beyond batch and contracting.
///
/// For lhs cotangent: contracts output cotangent with rhs over the N dim.
/// For rhs cotangent: contracts \(\mathrm{lhs}^{\mathsf{T}}\) with the output
/// cotangent over the \(M\) dimension.
/// Results are transposed from canonical [batch..., free, contracted] order
///  back to the original operand dim order via `transpose_to_match`.
///
/// Returns null if preconditions aren't met (no batch dims, multiple
///  contracting dims, or more than one free dim per operand).
fn maybe_batched_matmul_vjp(
    ctx: types.VjpContext,
    lhs_var: *const pr.Var,
    rhs_var: *const pr.Var,
    out_cot: *pr.Var,
    lhs_primal: *pr.Var,
    rhs_primal: *pr.Var,
    params: pr.DotGeneralParams,
) types.AdError!?BatchedMatmulVjpPair {
    const batch_len: usize = params.lhs_batch_dims.len;
    if (batch_len == 0) return null;
    if (batch_len != params.rhs_batch_dims.len) return null;
    if (params.lhs_contracting_dims.len != 1) return null;
    if (params.rhs_contracting_dims.len != 1) return null;

    const lhs_t = lhs_var.as_tensor();
    const rhs_t = rhs_var.as_tensor();
    const lhs_rank = lhs_t.shape.rank();
    const rhs_rank = rhs_t.shape.rank();
    if (lhs_rank < 2 or rhs_rank < 2) return null;

    // Identify the single free dim per operand: M for lhs, N for rhs.
    // Each operand is [batch..., M/N, K] (in some permutation).
    const lhs_k_dim: i64 = params.lhs_contracting_dims[0];
    const rhs_k_dim: i64 = params.rhs_contracting_dims[0];
    const lhs_m_dim = find_single_other_dim(lhs_rank, params.lhs_batch_dims, lhs_k_dim) orelse return null;
    const rhs_n_dim = find_single_other_dim(rhs_rank, params.rhs_batch_dims, rhs_k_dim) orelse return null;

    // Output layout is [batch..., M, N]. M is at position batch_len, N at batch_len+1.
    const out_m_dim: i64 = @intCast(batch_len);
    const out_n_dim: i64 = @intCast(batch_len + 1);

    var out_batch_buf: [8]i64 = undefined;
    if (batch_len > out_batch_buf.len) return error.UnsupportedDerivative;
    for (0..batch_len) |i| out_batch_buf[i] = @intCast(i);
    const out_batch = out_batch_buf[0..batch_len];

    // d_lhs = out_cot @ rhs^T: contract over N (output's rhs-free dim) with rhs's N dim.
    // The canonical [batch..., M, K] result is permuted to the lhs layout.
    var out_contract_n: [1]i64 = .{out_n_dim};
    var rhs_contract_n: [1]i64 = .{rhs_n_dim};
    const d_lhs_canon = try ctx.builder.dot_general(out_cot, rhs_primal, .{
        .lhs_batch_dims = out_batch,
        .rhs_batch_dims = params.rhs_batch_dims,
        .lhs_contracting_dims = out_contract_n[0..],
        .rhs_contracting_dims = rhs_contract_n[0..],
    });

    const d_lhs = try transpose_to_match(ctx.builder, d_lhs_canon, lhs_rank, params.lhs_batch_dims, lhs_m_dim, lhs_k_dim, batch_len);

    // d_rhs = lhs^T @ out_cot: contract over M (lhs's free dim) with output's M dim.
    // The canonical [batch..., K, N] result is permuted to the rhs layout.
    var lhs_contract_m: [1]i64 = .{lhs_m_dim};
    var out_contract_m: [1]i64 = .{out_m_dim};
    const d_rhs_canon = try ctx.builder.dot_general(lhs_primal, out_cot, .{
        .lhs_batch_dims = params.lhs_batch_dims,
        .rhs_batch_dims = out_batch,
        .lhs_contracting_dims = lhs_contract_m[0..],
        .rhs_contracting_dims = out_contract_m[0..],
    });

    const d_rhs = try transpose_to_match(ctx.builder, d_rhs_canon, rhs_rank, params.rhs_batch_dims, rhs_k_dim, rhs_n_dim, batch_len);

    return .{ .lhs = d_lhs, .rhs = d_rhs };
}

/// Find the single dimension that is neither a batch dim nor the contracting dim.
/// Returns null if there isn't exactly one such dimension.
fn find_single_other_dim(rank: usize, batch_dims: []const i64, contracting_dim: i64) ?i64 {
    var found: ?i64 = null;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        if (dim_i64 == contracting_dim) continue;
        if (index_of_i64(batch_dims, dim_i64) != null) continue;
        if (found != null) return null;
        found = dim_i64;
    }
    return found;
}

/// Collect all dimensions that are neither batch nor contracting ("other"/free dims).
fn collect_other_dims(
    allocator: std.mem.Allocator,
    rank: usize,
    batch_dims: []const i64,
    contracting_dims: []const i64,
) ![]i64 {
    var list = try std.ArrayList(i64).initCapacity(allocator, rank);
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        if (index_of_i64(batch_dims, dim_i64) != null) continue;
        if (index_of_i64(contracting_dims, dim_i64) != null) continue;
        list.appendAssumeCapacity(dim_i64);
    }
    return try list.toOwnedSlice(allocator);
}

fn build_range(allocator: std.mem.Allocator, start: usize, len: usize) ![]i64 {
    const out = try allocator.alloc(i64, len);
    for (0..len) |i| out[i] = @intCast(start + i);
    return out;
}

fn index_of_i64(list: []const i64, needle: i64) ?usize {
    for (list, 0..) |v, i| if (v == needle) return i;
    return null;
}

/// Build a permutation that maps canonical VJP output dims back to the original
///  operand layout. Canonical order is [batch..., a_dim, b_dim], this inverts
///  that mapping. Skips the transpose if the permutation is identity.
fn transpose_to_match(
    b: *pr.FunctionBuilder,
    canon: *pr.Var,
    rank: usize,
    batch_dims: []const i64,
    a_dim: i64,
    b_dim: i64,
    batch_len: usize,
) types.AdError!*pr.Var {
    var perm_buf: [8]i64 = undefined;
    if (rank > perm_buf.len) return error.UnsupportedDerivative;

    // For each target dim d, find where it sits in the canonical layout:
    //   batch dims : positions [0..batch_len)
    //   a_dim      : position batch_len
    //   b_dim      : position batch_len+1
    var is_identity = true;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        const src: i64 = if (index_of_i64(batch_dims, dim_i64)) |bi| blk: {
            break :blk @intCast(bi);
        } else if (dim_i64 == a_dim) @intCast(batch_len) else if (dim_i64 == b_dim) @intCast(batch_len + 1) else return error.UnsupportedDerivative;
        perm_buf[d] = src;
        if (src != dim_i64) is_identity = false;
    }

    if (is_identity) return canon;
    return try b.transpose(canon, perm_buf[0..rank]);
}

/// Multi-dim variant of `transpose_to_match`. Maps canonical
///  [batch..., other..., contracting...] order back to the original operand
///  layout by looking up each target dimension in the batch, other, and
///  contracting sets. Skips the transpose if the permutation is identity.
fn transpose_to_match_multi(
    ctx: types.VjpContext,
    canon: *pr.Var,
    rank: usize,
    batch_dims: []const i64,
    other_dims: []const i64,
    contract_dims: []const i64,
) types.AdError!*pr.Var {
    if (rank == 0) return canon;

    const perm = try ctx.allocator.alloc(i64, rank);
    defer ctx.allocator.free(perm);

    // For each target dim d, find where it sits in the canonical layout:
    //   batch dims     : positions [0..batch_len)
    //   other dims     : positions [batch_len..batch_len+other_len)
    //   contract dims  : positions [batch_len+other_len..)
    var is_identity = true;
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        const dim_i64: i64 = @intCast(d);
        const src: i64 = if (index_of_i64(batch_dims, dim_i64)) |bi| blk: {
            break :blk @intCast(bi);
        } else if (index_of_i64(other_dims, dim_i64)) |oi| blk: {
            break :blk @intCast(batch_dims.len + oi);
        } else if (index_of_i64(contract_dims, dim_i64)) |ci| blk: {
            break :blk @intCast(batch_dims.len + other_dims.len + ci);
        } else return error.UnsupportedDerivative;
        perm[d] = src;
        if (src != dim_i64) is_identity = false;
    }

    if (is_identity) return canon;
    return try ctx.builder.transpose(canon, perm);
}

const max_rank = pr.max_rank;

fn compute_convolution_output_dims(
    lhs: pr.Tensor,
    rhs: pr.Tensor,
    params: pr.ConvolutionParams,
    out_buf: *[max_rank]i64,
) ?[]const i64 {
    const rank = lhs.shape.rank();
    if (rank != rhs.shape.rank() or rank < 2 or rank > max_rank) return null;
    const spatial_rank = rank - 2;
    if (params.window_strides.len != spatial_rank or
        params.padding.len != spatial_rank * 2 or
        params.lhs_dilation.len != spatial_rank or
        params.rhs_dilation.len != spatial_rank or
        params.window_reversal.len != spatial_rank or
        params.dimensions.input_spatial_dimensions.len != spatial_rank or
        params.dimensions.kernel_spatial_dimensions.len != spatial_rank or
        params.dimensions.output_spatial_dimensions.len != spatial_rank)
    {
        return null;
    }
    if (!valid_dimension_spec(rank, params.dimensions.input_batch_dimension, params.dimensions.input_feature_dimension, params.dimensions.input_spatial_dimensions) or
        !valid_dimension_spec(rank, params.dimensions.kernel_output_feature_dimension, params.dimensions.kernel_input_feature_dimension, params.dimensions.kernel_spatial_dimensions) or
        !valid_dimension_spec(rank, params.dimensions.output_batch_dimension, params.dimensions.output_feature_dimension, params.dimensions.output_spatial_dimensions))
    {
        return null;
    }
    if (params.feature_group_count <= 0 or params.batch_group_count <= 0) return null;
    if (params.feature_group_count > 1 and params.batch_group_count > 1) return null;

    const input_batch: usize = @intCast(params.dimensions.input_batch_dimension);
    const input_feature: usize = @intCast(params.dimensions.input_feature_dimension);
    const kernel_input_feature: usize = @intCast(params.dimensions.kernel_input_feature_dimension);
    const kernel_output_feature: usize = @intCast(params.dimensions.kernel_output_feature_dimension);
    const feature_groups: i64 = params.feature_group_count;
    const batch_groups: i64 = params.batch_group_count;
    if (@mod(lhs.shape.dims[input_feature], feature_groups) != 0 or
        @divExact(lhs.shape.dims[input_feature], feature_groups) != rhs.shape.dims[kernel_input_feature] or
        @mod(rhs.shape.dims[kernel_output_feature], feature_groups) != 0 or
        @mod(lhs.shape.dims[input_batch], batch_groups) != 0 or
        @mod(rhs.shape.dims[kernel_output_feature], batch_groups) != 0)
    {
        return null;
    }

    @memset(out_buf[0..rank], 0);
    out_buf[@intCast(params.dimensions.output_batch_dimension)] = @divExact(lhs.shape.dims[input_batch], batch_groups);
    out_buf[@intCast(params.dimensions.output_feature_dimension)] = @divExact(rhs.shape.dims[kernel_output_feature], batch_groups);
    for (0..spatial_rank) |i| {
        const stride = params.window_strides[i];
        const lhs_dilation = params.lhs_dilation[i];
        const rhs_dilation = params.rhs_dilation[i];
        if (stride <= 0 or lhs_dilation <= 0 or rhs_dilation <= 0) return null;
        const lhs_size = lhs.shape.dims[@intCast(params.dimensions.input_spatial_dimensions[i])];
        const rhs_size = rhs.shape.dims[@intCast(params.dimensions.kernel_spatial_dimensions[i])];
        if (lhs_size <= 0 or rhs_size <= 0) return null;
        const dilated_lhs = (lhs_size - 1) * lhs_dilation + 1;
        const dilated_rhs = (rhs_size - 1) * rhs_dilation + 1;
        const numerator = dilated_lhs + params.padding[i * 2] + params.padding[i * 2 + 1] - dilated_rhs;
        if (numerator < 0) return null;
        out_buf[@intCast(params.dimensions.output_spatial_dimensions[i])] = @divFloor(numerator, stride) + 1;
    }
    return out_buf[0..rank];
}

fn valid_dimension_spec(rank: usize, first: i64, second: i64, spatial: []const i64) bool {
    var seen = @as([max_rank]bool, @splat(false));
    for ([_]i64{ first, second }) |dim| {
        if (dim < 0 or dim >= rank or seen[@intCast(dim)]) return false;
        seen[@intCast(dim)] = true;
    }
    for (spatial) |dim| {
        if (dim < 0 or dim >= rank or seen[@intCast(dim)]) return false;
        seen[@intCast(dim)] = true;
    }
    for (seen[0..rank]) |present| if (!present) return false;
    return true;
}

fn dilated_size(size: i64, dilation: i64) i64 {
    return (size - 1) * dilation + 1;
}

/// Compute `dot_general` output dimensions.
///
/// `output = [batch_dims..., lhs_other_dims..., rhs_other_dims...]`
///
/// Validates that batch dims agree in size, contracting dims agree in size, and
///  no dimension is claimed by both batch and contracting sets.
/// Returns null if parameters are invalid.
/// Result is a slice into `out_buf`, caller must dupe if the data needs to outlive the buffer.
fn compute_dot_general_output_dims(
    lhs: pr.Tensor,
    rhs: pr.Tensor,
    params: pr.DotGeneralParams,
    out_buf: *[max_rank]i64,
) ?[]const i64 {
    const lhs_rank = lhs.shape.rank();
    const rhs_rank = rhs.shape.rank();
    if (lhs_rank > max_rank or rhs_rank > max_rank) return null;

    if (params.lhs_batch_dims.len != params.rhs_batch_dims.len) return null;
    if (params.lhs_contracting_dims.len != params.rhs_contracting_dims.len) return null;

    // Phase 1: Mark and validate batch dims. Paired dims must have matching sizes.
    var lhs_batch = @as([max_rank]bool, @splat(false));
    var rhs_batch = @as([max_rank]bool, @splat(false));
    for (params.lhs_batch_dims, 0..) |d, i| {
        if (d < 0) return null;
        const lhs_idx: usize = @intCast(d);
        if (lhs_idx >= lhs_rank or lhs_batch[lhs_idx]) return null;
        const rhs_d = params.rhs_batch_dims[i];
        if (rhs_d < 0) return null;
        const rhs_idx: usize = @intCast(rhs_d);
        if (rhs_idx >= rhs_rank or rhs_batch[rhs_idx]) return null;
        if (lhs.shape.dims[lhs_idx] != rhs.shape.dims[rhs_idx]) return null;
        lhs_batch[lhs_idx] = true;
        rhs_batch[rhs_idx] = true;
    }

    // Phase 2: Mark and validate contracting dims. Must not overlap with batch dims.
    var lhs_contract = @as([max_rank]bool, @splat(false));
    var rhs_contract = @as([max_rank]bool, @splat(false));
    for (params.lhs_contracting_dims, 0..) |d, i| {
        if (d < 0) return null;
        const lhs_idx: usize = @intCast(d);
        if (lhs_idx >= lhs_rank or lhs_contract[lhs_idx] or lhs_batch[lhs_idx]) return null;
        const rhs_d = params.rhs_contracting_dims[i];
        if (rhs_d < 0) return null;
        const rhs_idx: usize = @intCast(rhs_d);
        if (rhs_idx >= rhs_rank or rhs_contract[rhs_idx] or rhs_batch[rhs_idx]) return null;
        if (lhs.shape.dims[lhs_idx] != rhs.shape.dims[rhs_idx]) return null;
        lhs_contract[lhs_idx] = true;
        rhs_contract[rhs_idx] = true;
    }

    // Phase 3: Assemble output dims in StableHLO order:
    //   [batch sizes, lhs "other" sizes, rhs "other" sizes]
    const out_rank = params.lhs_batch_dims.len +
        (lhs_rank - params.lhs_batch_dims.len - params.lhs_contracting_dims.len) +
        (rhs_rank - params.rhs_batch_dims.len - params.rhs_contracting_dims.len);
    if (out_rank > max_rank) return null;

    var out_i: usize = 0;
    for (params.lhs_batch_dims) |d| {
        out_buf[out_i] = lhs.shape.dims[@intCast(d)];
        out_i += 1;
    }
    for (0..lhs_rank) |i| {
        if (lhs_batch[i] or lhs_contract[i]) continue;
        out_buf[out_i] = lhs.shape.dims[i];
        out_i += 1;
    }
    for (0..rhs_rank) |i| {
        if (rhs_batch[i] or rhs_contract[i]) continue;
        out_buf[out_i] = rhs.shape.dims[i];
        out_i += 1;
    }
    return out_buf[0..out_i];
}

fn write_dims(writer: *types.Writer, dims: []const i64) types.FormatError!void {
    try writer.writeAll("[");
    for (dims, 0..) |d, i| {
        if (i != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{d});
    }
    try writer.writeAll("]");
}
