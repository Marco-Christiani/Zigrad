//! Shape Operations
//! Ops that manipulate tensor shape without changing element values.
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const log = std.log.scoped(.@"zg/shape");
const Tensor = pr.Tensor;
const Aval = pr.Aval;
const max_rank = pr.max_rank;

// Reshape

pub const reshape = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, rp: pr.ReshapeParams) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();

        if (!std.mem.eql(i64, out.shape.dims, rp.out_shape)) {
            log.err(
                "reshape out dims mismatch: operand={any} out={any} param_out={any}",
                .{ operand.shape.dims, out.shape.dims, rp.out_shape },
            );
            return error.ReshapeTypeMismatch;
        }
        if (operand.dtype != out.dtype) {
            log.err(
                "reshape dtype mismatch: operand={s} out={s}",
                .{ @tagName(operand.dtype), @tagName(out.dtype) },
            );
            return error.ReshapeTypeMismatch;
        }
        if (num_elements(operand.shape.dims) != num_elements(out.shape.dims)) {
            log.err(
                "reshape element count mismatch: operand={any} out={any}",
                .{ operand.shape.dims, out.shape.dims },
            );
            return error.ReshapeTypeMismatch;
        }
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, rp: pr.ReshapeParams) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;

        const operand = inputs[0].as_tensor();
        if (num_elements(operand.shape.dims) != num_elements(rp.out_shape)) return error.ReshapeTypeMismatch;
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = rp.out_shape } } };
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, _: pr.ReshapeParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const operand_tensor = op.operand(0).as_tensor();

        const contrib = try ctx.builder.reshape(out_cot, operand_tensor.shape.dims);
        try ctx.add_cotangent(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{reshape}(x, s)) =
    ///  \operatorname{reshape}(\mathrm{d}x, s)\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, _: pr.ReshapeParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const dx = try ctx.tangent_or_zero(op.operand(0));
        const out_tensor = op.result(0).as_tensor();
        ctx.set_tangent(op.result(0), try ctx.builder.reshape(dx, out_tensor.shape.dims));
    }

    pub fn format(writer: *types.Writer, op: *const pr.Op, rp: pr.ReshapeParams) types.FormatError!void {
        if (op.inputs.len > 0) {
            const src = op.operand(0).as_tensor();
            try format_shape(writer, src.shape.dims);
        }
        try writer.writeAll(" -> ");
        try format_shape(writer, rp.out_shape);
    }
};

// Iota

pub const iota = struct {
    pub const arity = .{ .in = 0, .out = 1 };

    pub fn validate(op: *const pr.Op, ip: pr.IotaParams) pr.ValidationError!void {
        if (op.inputs.len != 0 or op.outputs.len != 1) return error.InvalidOpArity;

        const out = op.result(0).as_tensor();

        if (!std.mem.eql(i64, out.shape.dims, ip.out_shape)) {
            log.err("iota out dims mismatch: out={any} param_out={any}", .{ out.shape.dims, ip.out_shape });
            return error.IotaTypeMismatch;
        }
        if (out.dtype != ip.out_dtype) {
            log.err("iota dtype mismatch: out={s} param_out={s}", .{ @tagName(out.dtype), @tagName(ip.out_dtype) });
            return error.IotaTypeMismatch;
        }
        if (ip.out_dtype != .i32 and ip.out_dtype != .i64) {
            log.err("iota dtype must be integer: dtype={s}", .{@tagName(ip.out_dtype)});
            return error.IotaTypeMismatch;
        }
        if (ip.dimension < 0 or @as(usize, @intCast(ip.dimension)) >= out.shape.rank()) {
            log.err("iota dimension out of range: dim={d} rank={d}", .{ ip.dimension, out.shape.rank() });
            return error.IotaTypeMismatch;
        }
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, ip: pr.IotaParams) pr.BuildError!Aval {
        if (inputs.len != 0) return error.InvalidOpArity;
        if (ip.out_dtype != .i32 and ip.out_dtype != .i64) return error.IotaTypeMismatch;
        if (ip.dimension < 0 or @as(usize, @intCast(ip.dimension)) >= ip.out_shape.len) return error.IotaTypeMismatch;
        return .{ .tensor = .{ .dtype = ip.out_dtype, .shape = .{ .dims = ip.out_shape } } };
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, ip: pr.IotaParams) types.FormatError!void {
        try writer.print("dim={d} dtype={s} shape=", .{ ip.dimension, @tagName(ip.out_dtype) });
        try format_shape(writer, ip.out_shape);
    }
};

// Transpose

pub const transpose = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, tp: pr.TransposeParams) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();

        if (operand.dtype != out.dtype) return error.TransposeTypeMismatch;
        if (!is_permutation(tp.permutation, operand.shape.rank())) return error.TransposeTypeMismatch;
        if (out.shape.rank() != operand.shape.rank()) return error.TransposeTypeMismatch;

        for (tp.permutation, 0..) |p, out_axis| {
            const in_axis: usize = @intCast(p);
            if (out.shape.dims[out_axis] != operand.shape.dims[in_axis]) return error.TransposeTypeMismatch;
        }
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, tp: pr.TransposeParams) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;

        const operand = inputs[0].as_tensor();
        if (!is_permutation(tp.permutation, operand.shape.rank())) return error.TransposeTypeMismatch;

        const out_dims = try alloc.alloc(i64, operand.shape.rank());
        for (tp.permutation, 0..) |p, i| out_dims[i] = operand.shape.dims[@intCast(p)];
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, tp: pr.TransposeParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;

        // Inverse permutation
        const inv = try ctx.allocator.alloc(i64, tp.permutation.len);
        defer ctx.allocator.free(inv);
        for (tp.permutation, 0..) |p, i| inv[@intCast(p)] = @intCast(i);

        const contrib = try ctx.builder.transpose(out_cot, inv);
        try ctx.add_cotangent(op.operand(0), contrib);
    }

    /// JVP: \(\mathrm{d}(\operatorname{transpose}(x, \mathrm{perm})) =
    ///  \operatorname{transpose}(\mathrm{d}x, \mathrm{perm})\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, tp: pr.TransposeParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const dx = try ctx.tangent_or_zero(op.operand(0));
        ctx.set_tangent(op.result(0), try ctx.builder.transpose(dx, tp.permutation));
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, tp: pr.TransposeParams) types.FormatError!void {
        try writer.writeAll("perm=[");
        for (tp.permutation, 0..) |p, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{p});
        }
        try writer.writeByte(']');
    }
};

// Slice

pub const slice = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, sparams: pr.SliceParams) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;
        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();
        if (!slice_matches(operand.shape.dims, out.shape.dims, sparams)) return error.SliceTypeMismatch;
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, sparams: pr.SliceParams) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;
        const operand = inputs[0].as_tensor();
        const out_dims = try compute_slice_output_dims(alloc, operand.shape.dims, sparams);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    /// JVP: \(\mathrm{d}(\operatorname{slice}(x, p)) =
    ///  \operatorname{slice}(\mathrm{d}x, p)\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, sparams: pr.SliceParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const dx = try ctx.tangent_or_zero(op.operand(0));
        ctx.set_tangent(op.result(0), try ctx.builder.slice(dx, sparams));
    }

    /// VJP backward for slice. Reconstructs the full input cotangent by
    /// zero-padding the slice cotangent back to the original shape. For each
    /// axis, prepends `start` zeros and appends `input_dim - limit` zeros via
    /// concatenation. Only supports unit strides.
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, sparams: pr.SliceParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const in_tensor = op.operand(0).as_tensor();

        for (sparams.strides) |s| {
            if (s != 1) return error.UnsupportedDerivative;
        }

        // Build the full input cotangent by padding each axis with zeros.
        // For axis i with slice [start:limit], prepend `start` zeros and
        // append `input_dim - limit` zeros via concatenation.
        var cur = out_cot;
        var cur_dims = op.result(0).as_tensor().shape.dims;

        var axis: usize = 0;
        while (axis < in_tensor.shape.rank()) : (axis += 1) {
            const start = sparams.start_indices[axis];
            const limit = sparams.limit_indices[axis];
            if (start < 0 or limit < 0) return error.UnsupportedDerivative;
            const pre: i64 = start; // zeros before the slice
            const post: i64 = in_tensor.shape.dims[axis] - limit; // zeros after the slice

            if (pre > 0) {
                const pre_dims = try ctx.allocator.dupe(i64, cur_dims);
                defer ctx.allocator.free(pre_dims);
                pre_dims[axis] = pre;
                const pre_tensor = try ctx.builder.scalar_broadcast(in_tensor.dtype, pre_dims, 0.0);
                const cat = try ctx.builder.concatenate(&.{ pre_tensor, cur }, @intCast(axis));
                cur = cat;
                cur_dims = cur.as_tensor().shape.dims;
            }

            if (post > 0) {
                const post_dims = try ctx.allocator.dupe(i64, cur_dims);
                defer ctx.allocator.free(post_dims);
                post_dims[axis] = post;
                const post_tensor = try ctx.builder.scalar_broadcast(in_tensor.dtype, post_dims, 0.0);
                const cat = try ctx.builder.concatenate(&.{ cur, post_tensor }, @intCast(axis));
                cur = cat;
                cur_dims = cur.as_tensor().shape.dims;
            }
        }

        try ctx.add_cotangent(op.operand(0), cur);
    }
};

// Concatenate

pub const concatenate = struct {
    pub const arity = .{ .in = .any, .out = 1 };

    pub fn validate(op: *const pr.Op, cp: pr.ConcatenateParams) pr.ValidationError!void {
        if (op.inputs.len == 0 or op.outputs.len != 1) return error.InvalidOpArity;
        const out = op.result(0).as_tensor();
        if (!concat_matches_op(op, out, cp.axis)) return error.ConcatTypeMismatch;
        const first = op.operand(0).as_tensor();
        if (first.dtype != out.dtype) return error.ConcatTypeMismatch;
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, cp: pr.ConcatenateParams) pr.BuildError!Aval {
        if (inputs.len == 0) return error.InvalidOpArity;
        const first = inputs[0].as_tensor();
        const out_dims = try compute_concat_output_dims(alloc, inputs, cp.axis);
        return .{ .tensor = .{ .dtype = first.dtype, .shape = .{ .dims = out_dims } } };
    }

    /// JVP: \(\mathrm{d}(\operatorname{concatenate}(x_s, \mathrm{axis})) =
    ///  \operatorname{concatenate}(\mathrm{d}x_s, \mathrm{axis})\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, cp: pr.ConcatenateParams) types.AdError!void {
        if (op.inputs.len == 0) return error.InvalidOpArity;

        var tangents = try ctx.allocator.alloc(*pr.Var, op.inputs.len);
        defer ctx.allocator.free(tangents);
        for (op.inputs, 0..) |operand, i| {
            tangents[i] = try ctx.tangent_or_zero(operand.value);
        }
        ctx.set_tangent(op.result(0), try ctx.builder.concatenate(tangents, cp.axis));
    }

    /// VJP backward for concatenate. Slices the concatenated cotangent back
    /// into per-input contributions. Tracks a running offset along the concat
    /// axis and extracts each input's slice at [offset, offset+len).
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, cp: pr.ConcatenateParams) types.AdError!void {
        if (op.inputs.len == 0) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const axis_u: usize = @intCast(cp.axis);

        // Walk the concat axis, extracting each input's slice from the cotangent.
        // offset tracks the running position along the concat axis.
        var offset: i64 = 0;
        for (op.inputs) |operand| {
            const t = operand.value.as_tensor();
            const len = t.shape.dims[axis_u];
            const start = offset;
            const limit = offset + len;

            // Build slice params: full range on all axes except the concat axis,
            // where we take [offset, offset+len).
            var start_indices = try ctx.allocator.alloc(i64, t.shape.rank());
            var limit_indices = try ctx.allocator.alloc(i64, t.shape.rank());
            var strides = try ctx.allocator.alloc(i64, t.shape.rank());
            defer ctx.allocator.free(start_indices);
            defer ctx.allocator.free(limit_indices);
            defer ctx.allocator.free(strides);

            for (t.shape.dims, 0..) |d, i| {
                start_indices[i] = 0;
                limit_indices[i] = @intCast(d);
                strides[i] = 1;
            }
            start_indices[axis_u] = start;
            limit_indices[axis_u] = limit;

            const slice_out = try ctx.builder.slice(out_cot, .{
                .start_indices = start_indices,
                .limit_indices = limit_indices,
                .strides = strides,
            });
            try ctx.add_cotangent(operand.value, slice_out);
            offset += len;
        }
    }
};

// Reduce

pub const reduce = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, rp: pr.ReduceParams) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();

        if (!reduce_output_matches(operand.shape.dims, out.shape.dims, rp.axes))
            return error.ReduceTypeMismatch;
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, rp: pr.ReduceParams) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;
        const operand = inputs[0].as_tensor();
        const out_dims = try reduce_output_dims(alloc, operand.shape.dims, rp.axes);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, rp: pr.ReduceParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;
        switch (rp.operation) {
            .sum => {
                const dx = try ctx.tangent_or_zero(op.operand(0));
                ctx.set_tangent(op.result(0), try ctx.builder.reduce(dx, rp));
            },
            .maximum => try maximum_jvp(ctx, op, rp),
        }
    }

    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, rp: pr.ReduceParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;
        switch (rp.operation) {
            .sum => try sum_vjp(ctx, op, rp),
            .maximum => try maximum_vjp(ctx, op, rp),
        }
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, rp: pr.ReduceParams) types.FormatError!void {
        try writer.print("operation={t}, axes=[", .{rp.operation});
        for (rp.axes, 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeByte(']');
    }
};

fn maximum_jvp(ctx: types.JvpContext, op: *const pr.Op, rp: pr.ReduceParams) types.AdError!void {
    const dx = try ctx.tangent_or_zero(op.operand(0));
    const operand = try ctx.primal(op.operand(0));
    const out_primal = try ctx.primal(op.result(0));
    const in_tensor = op.operand(0).as_tensor();

    const bd = try reduce_broadcast_dims(ctx.allocator, in_tensor.shape.dims.len, rp.axes);
    defer ctx.allocator.free(bd);

    const max_b = try ctx.builder.broadcast_in_dim(out_primal, in_tensor.shape.dims, bd);
    const cmp_type = compare_type_for_dtype(in_tensor.dtype);
    const mask = try ctx.builder.compare(operand, max_b, .{
        .direction = .EQ,
        .compare_type = cmp_type,
    });
    const mask_f = try ctx.builder.convert(mask, in_tensor.dtype);
    const masked_dx = try ctx.builder.multiply(mask_f, dx);
    ctx.set_tangent(op.result(0), try ctx.builder.reduce(masked_dx, .{
        .axes = rp.axes,
        .operation = .sum,
    }));
}

fn maximum_vjp(ctx: types.VjpContext, op: *const pr.Op, rp: pr.ReduceParams) types.AdError!void {
    const out_cot = ctx.cotangent(op.result(0)) orelse return;
    const in_tensor = op.operand(0).as_tensor();

    const bd = try reduce_broadcast_dims(ctx.allocator, in_tensor.shape.dims.len, rp.axes);
    defer ctx.allocator.free(bd);

    // Broadcast the reduced max and cotangent back to the full input shape.
    const out_primal = try ctx.primal(op.result(0));
    const max_b = try ctx.builder.broadcast_in_dim(out_primal, in_tensor.shape.dims, bd);
    const out_cot_b = try ctx.builder.broadcast_in_dim(out_cot, in_tensor.shape.dims, bd);

    // Build a boolean mask: true where input == max (these positions contributed to the max).
    const cmp_type = compare_type_for_dtype(in_tensor.dtype);
    const operand = try ctx.primal(op.operand(0));
    const mask = try ctx.builder.compare(operand, max_b, .{
        .direction = .EQ,
        .compare_type = cmp_type,
    });
    // Convert bool mask to input dtype and multiply with cotangent.
    // Positions not equal to max get zero cotangent.
    const mask_f = try ctx.builder.convert(mask, in_tensor.dtype);
    const contrib = try ctx.builder.multiply(out_cot_b, mask_f);
    try ctx.add_cotangent(op.operand(0), contrib);
}

fn sum_vjp(ctx: types.VjpContext, op: *const pr.Op, rp: pr.ReduceParams) types.AdError!void {
    const out_cot = ctx.cotangent(op.result(0)) orelse return;
    const in_tensor = op.operand(0).as_tensor();
    const dimensions = try reduce_broadcast_dims(ctx.allocator, in_tensor.shape.dims.len, rp.axes);
    defer ctx.allocator.free(dimensions);
    const contribution = try ctx.builder.broadcast_in_dim(
        out_cot,
        in_tensor.shape.dims,
        dimensions,
    );
    try ctx.add_cotangent(op.operand(0), contribution);
}

// Gather

pub const gather = struct {
    pub const arity = .{ .in = 2, .out = 1 };

    pub fn validate(op: *const pr.Op, gparams: pr.GatherParams) pr.ValidationError!void {
        if (op.inputs.len != 2 or op.outputs.len != 1) return error.InvalidOpArity;
        const operand = op.operand(0).as_tensor();
        const indices = op.operand(1).as_tensor();
        const out = op.result(0).as_tensor();

        if (out.dtype != operand.dtype) {
            log.err("gather dtype mismatch: operand={s} out={s}", .{ @tagName(operand.dtype), @tagName(out.dtype) });
            return error.GatherTypeMismatch;
        }
        if (indices.dtype != .i32 and indices.dtype != .i64) {
            log.err("gather indices dtype mismatch: indices={s}", .{@tagName(indices.dtype)});
            return error.GatherTypeMismatch;
        }
        var dims_buf: [max_rank]i64 = undefined;
        const expected = compute_gather_output_dims(operand.shape.dims, indices.shape.dims, gparams, &dims_buf) orelse {
            log.err(
                "gather output shape invalid: operand={any} indices={any} params(slice={any}, offset={any}, collapsed={any}, map={any}, index_vec_dim={d})",
                .{
                    operand.shape.dims,
                    indices.shape.dims,
                    gparams.slice_sizes,
                    gparams.offset_dims,
                    gparams.collapsed_slice_dims,
                    gparams.start_index_map,
                    gparams.index_vector_dim,
                },
            );
            return error.GatherTypeMismatch;
        };
        if (!std.mem.eql(i64, out.shape.dims, expected)) {
            log.err(
                "gather out dims mismatch: operand={any} indices={any} out={any} expected={any}",
                .{ operand.shape.dims, indices.shape.dims, out.shape.dims, expected },
            );
            return error.GatherTypeMismatch;
        }
    }

    pub fn infer_output(alloc: std.mem.Allocator, inputs: []const *pr.Var, gparams: pr.GatherParams) pr.BuildError!Aval {
        if (inputs.len != 2) return error.InvalidOpArity;
        const operand = inputs[0].as_tensor();
        const indices = inputs[1].as_tensor();
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.GatherTypeMismatch;
        var dims_buf: [max_rank]i64 = undefined;
        const computed = compute_gather_output_dims(operand.shape.dims, indices.shape.dims, gparams, &dims_buf) orelse
            return error.GatherTypeMismatch;
        const out_dims = try alloc.dupe(i64, computed);
        return .{ .tensor = .{ .dtype = operand.dtype, .shape = .{ .dims = out_dims } } };
    }

    /// JVP: \(\mathrm{d}(\operatorname{gather}(x, \mathrm{idx}, p)) =
    ///  \operatorname{gather}(\mathrm{d}x, \mathrm{idx}, p)\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, gparams: pr.GatherParams) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const dx = try ctx.tangent_or_zero(op.operand(0));
        const indices = try ctx.primal(op.operand(1));
        ctx.set_tangent(op.result(0), try ctx.builder.gather(dx, indices, gparams));
    }

    /// VJP backward for gather.
    ///
    /// Inverts the gather by scatter-adding the output cotangent into a zero
    ///  tensor of the original operand shape.
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, gparams: pr.GatherParams) types.AdError!void {
        if (op.inputs.len != 2) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const indices = try ctx.primal(op.operand(1));
        const operand_tensor = op.operand(0).as_tensor();

        // zero tensor matching the original operand shape
        const zero = try ctx.builder.scalar(operand_tensor.dtype, 0.0);
        const zero_full = try ctx.builder.broadcast_in_dim(zero, operand_tensor.shape.dims, &.{});

        // scatter-add the gathered cotangent back to the operand positions
        // scatter_params_for_gather inverts the gather dimension mapping
        const sparams = scatter_params_for_gather(gparams);
        const contrib = try ctx.builder.scatter(zero_full, indices, out_cot, sparams);
        try ctx.add_cotangent(op.operand(0), contrib);
    }
};

// Scatter Add

pub const scatter = struct {
    pub const arity = .{ .in = 3, .out = 1 };

    pub fn validate(op: *const pr.Op, _: pr.ScatterParams) pr.ValidationError!void {
        if (op.inputs.len != 3 or op.outputs.len != 1) return error.InvalidOpArity;
        const operand = op.operand(0).as_tensor();
        const indices = op.operand(1).as_tensor();
        const updates = op.operand(2).as_tensor();
        const out = op.result(0).as_tensor();

        if (!types.same_tensor_type(operand, out)) return error.ScatterAddTypeMismatch;
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.ScatterTypeMismatch;
        if (updates.dtype != operand.dtype) return error.ScatterTypeMismatch;
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, _: pr.ScatterParams) pr.BuildError!Aval {
        if (inputs.len != 3) return error.InvalidOpArity;
        const operand = inputs[0].as_tensor();
        const indices = inputs[1].as_tensor();
        if (indices.dtype != .i32 and indices.dtype != .i64) return error.ScatterTypeMismatch;
        const updates = inputs[2].as_tensor();
        if (updates.dtype != operand.dtype) return error.ScatterTypeMismatch;
        return .{ .tensor = operand };
    }
};

// Broadcast In Dim

pub const broadcast_in_dim = struct {
    pub const arity = .{ .in = 1, .out = 1 };

    pub fn validate(op: *const pr.Op, bp: pr.BroadcastInDimParams) pr.ValidationError!void {
        if (op.inputs.len != 1 or op.outputs.len != 1) return error.InvalidOpArity;

        const operand = op.operand(0).as_tensor();
        const out = op.result(0).as_tensor();

        if (!std.mem.eql(i64, out.shape.dims, bp.out_shape)) return error.BroadcastInDimTypeMismatch;
        try validate_broadcast_in_dim_op(operand, out, bp.dimensions);
    }

    pub fn infer_output(_: std.mem.Allocator, inputs: []const *pr.Var, bp: pr.BroadcastInDimParams) pr.BuildError!Aval {
        if (inputs.len != 1) return error.InvalidOpArity;

        const operand = inputs[0].as_tensor();
        const out_tensor = Tensor{ .dtype = operand.dtype, .shape = .{ .dims = bp.out_shape } };

        try validate_broadcast_in_dim_op(operand, out_tensor, bp.dimensions);
        return .{ .tensor = out_tensor };
    }

    /// JVP: \(\mathrm{d}(\operatorname{broadcast\_in\_dim}(x, s, \mathrm{bd})) =
    ///  \operatorname{broadcast\_in\_dim}(\mathrm{d}x, s, \mathrm{bd})\).
    pub fn jvp(ctx: types.JvpContext, op: *const pr.Op, bp: pr.BroadcastInDimParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const dx = try ctx.tangent_or_zero(op.operand(0));
        ctx.set_tangent(op.result(0), try ctx.builder.broadcast_in_dim(dx, bp.out_shape, bp.dimensions));
    }

    /// VJP backward for broadcast_in_dim.
    ///
    /// Inverts the broadcast by summing along axes that were expanded
    ///  (size-1 -> size-N, or newly introduced).
    /// After reduction, reshapes to match the original input shape if the reduced shape differs
    ///  (e.g. when size-1 dims remain).
    pub fn vjp(ctx: types.VjpContext, op: *const pr.Op, bp: pr.BroadcastInDimParams) types.AdError!void {
        if (op.inputs.len != 1) return error.InvalidOpArity;

        const out_cot = ctx.cotangent(op.result(0)) orelse return;
        const in_tensor = op.operand(0).as_tensor();
        const out_tensor = op.result(0).as_tensor();

        // find axes that were expanded by the broadcast (size-1 -> N or newly introduced)
        const reduce_axes = try broadcast_reduce_axes(ctx.allocator, in_tensor, out_tensor, bp.dimensions);
        defer ctx.allocator.free(reduce_axes);

        // sum along those axes to collapse the broadcast back
        var contrib = out_cot;
        if (reduce_axes.len > 0) {
            contrib = try ctx.builder.reduce(contrib, .{ .axes = reduce_axes, .operation = .sum });
        }

        // after reduction shape may differ from input (e.g. size-1 dims that were
        //  broadcast still appear as size-1 in input but are dropped by reduction),
        //  so we reshape to match exactly.
        const reduced_dims = try reduce_output_dims(ctx.allocator, out_tensor.shape.dims, reduce_axes);
        defer ctx.allocator.free(reduced_dims);
        if (!std.mem.eql(i64, reduced_dims, in_tensor.shape.dims)) {
            contrib = try ctx.builder.reshape(contrib, in_tensor.shape.dims);
        }

        try ctx.add_cotangent(op.operand(0), contrib);
    }

    pub fn format(writer: *types.Writer, op: *const pr.Op, bp: pr.BroadcastInDimParams) types.FormatError!void {
        if (op.inputs.len > 0) {
            const src = op.operand(0).as_tensor();
            try format_shape(writer, src.shape.dims);
        }
        try writer.writeAll(" -> ");
        try format_shape(writer, bp.out_shape);
        try writer.writeAll(", dims=");
        try format_shape(writer, bp.dimensions);
    }
};

fn num_elements(dims: []const i64) usize {
    var n: usize = 1;
    for (dims) |d| n *= @intCast(d);
    return n;
}

fn is_permutation(perm: []const i64, rank: usize) bool {
    if (perm.len != rank) return false;
    if (rank == 0) return true;
    if (rank > max_rank) return false;
    var seen = @as([max_rank]bool, @splat(false));
    for (perm) |p| {
        if (p < 0) return false;
        const idx: usize = @intCast(p);
        if (idx >= rank) return false;
        if (seen[idx]) return false;
        seen[idx] = true;
    }
    return true;
}

// TODO(shape): Move common shape formatting onto `pr.Shape`.
fn format_shape(writer: *types.Writer, dims: []const i64) types.FormatError!void {
    try writer.writeByte('[');
    for (dims, 0..) |d, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("{d}", .{d});
    }
    try writer.writeByte(']');
}

fn validate_broadcast_in_dim_op(operand: Tensor, out: Tensor, broadcast_dimensions: []const i64) pr.ValidationError!void {
    if (operand.dtype != out.dtype) return error.BroadcastInDimTypeMismatch;
    if (broadcast_dimensions.len != operand.shape.rank()) return error.BroadcastInDimTypeMismatch;
    if (out.shape.rank() < operand.shape.rank()) return error.BroadcastInDimTypeMismatch;
    if (out.shape.rank() > max_rank) return error.BroadcastInDimTypeMismatch;
    var seen = @as([max_rank]bool, @splat(false));
    for (broadcast_dimensions, 0..) |d, i| {
        if (d < 0) return error.BroadcastInDimTypeMismatch;
        const out_dim_index: usize = @intCast(d);
        if (out_dim_index >= out.shape.rank()) return error.BroadcastInDimTypeMismatch;
        if (seen[out_dim_index]) return error.BroadcastInDimTypeMismatch;
        seen[out_dim_index] = true;
        const in_dim = operand.shape.dims[i];
        const out_dim = out.shape.dims[out_dim_index];
        if (in_dim != 1 and in_dim != out_dim) return error.BroadcastInDimTypeMismatch;
    }
}

/// Compute the output shape after reducing `axes` from `in_dims`.
///
/// Drops the reduced dimensions, preserving order of the remaining ones.
fn reduce_output_dims(allocator: std.mem.Allocator, in_dims: []const i64, axes: []const i64) pr.BuildError![]const i64 {
    const rank = in_dims.len;
    // TODO(pr): Give shared shape helpers errors independent of individual ops.
    if (rank > max_rank) return error.ReduceTypeMismatch;
    var reduced_axes = @as([max_rank]bool, @splat(false));
    for (axes) |axis| {
        if (axis < 0) return error.ReduceTypeMismatch;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return error.ReduceTypeMismatch;
        if (reduced_axes[idx]) return error.ReduceTypeMismatch;
        reduced_axes[idx] = true;
    }
    var out_count: usize = 0;
    for (0..rank) |i| {
        if (!reduced_axes[i]) out_count += 1;
    }
    const out_dims = try allocator.alloc(i64, out_count);
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduced_axes[i]) continue;
        out_dims[out_i] = in_dims[i];
        out_i += 1;
    }
    return out_dims;
}

fn reduce_output_matches(in_dims: []const i64, out_dims: []const i64, axes: []const i64) bool {
    const rank = in_dims.len;
    if (rank > max_rank) return false;
    var reduced_axes = @as([max_rank]bool, @splat(false));
    for (axes) |axis| {
        if (axis < 0) return false;
        const idx: usize = @intCast(axis);
        if (idx >= rank) return false;
        if (reduced_axes[idx]) return false;
        reduced_axes[idx] = true;
    }
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduced_axes[i]) continue;
        if (out_i >= out_dims.len) return false;
        if (in_dims[i] != out_dims[out_i]) return false;
        out_i += 1;
    }
    return out_i == out_dims.len;
}

/// Compute broadcast_dimensions for re-broadcasting a reduced result back to
///  the original shape.
///
/// Returns the indices of non-reduced dimensions, these map reduced-output
///  dims to their positions in the original rank.
fn reduce_broadcast_dims(allocator: std.mem.Allocator, rank: usize, axes: []const i64) pr.BuildError![]const i64 {
    if (rank > max_rank) return error.ReduceTypeMismatch;
    var reduced_axes = @as([max_rank]bool, @splat(false));
    for (axes) |axis| {
        const idx: usize = @intCast(axis);
        reduced_axes[idx] = true;
    }
    const keep = rank - axes.len;
    const bd = try allocator.alloc(i64, keep);
    var out_i: usize = 0;
    for (0..rank) |i| {
        if (reduced_axes[i]) continue;
        bd[out_i] = @intCast(i);
        out_i += 1;
    }
    return bd;
}

/// Determine which output axes must be reduced to invert a broadcast_in_dim.
///
/// An axis is reduced if: (1) it was a size-1 input dim broadcast to size-N,
///  or (2) it is an output dim not mapped by any broadcast dimension (newly
///  introduced).
///
///  Used by `broadcast_in_dim.vjp`.
fn broadcast_reduce_axes(
    allocator: std.mem.Allocator,
    in_tensor: Tensor,
    out_tensor: Tensor,
    bd: []const i64,
) pr.BuildError![]const i64 {
    if (out_tensor.shape.rank() > max_rank) return error.ReduceTypeMismatch;
    // track which output dims are mapped by a broadcast_dimension entry
    var mapped = @as([max_rank]bool, @splat(false));
    var reduced_axes = @as([max_rank]bool, @splat(false));
    for (bd, 0..) |d, i| {
        const out_idx: usize = @intCast(d);
        mapped[out_idx] = true;
        const in_dim = in_tensor.shape.dims[i];
        const out_dim = out_tensor.shape.dims[out_idx];
        // Case 1: input dim was 1 but output dim is >1 so this axis was broadcast-expanded.
        if (in_dim == 1 and out_dim > 1) {
            reduced_axes[out_idx] = true;
        }
    }
    // Case 2: output dims not referenced by any broadcast_dimension are newly introduced.
    for (0..out_tensor.shape.rank()) |i| {
        if (!mapped[i]) reduced_axes[i] = true;
    }
    var count: usize = 0;
    for (0..out_tensor.shape.rank()) |i| {
        if (reduced_axes[i]) count += 1;
    }
    const axes = try allocator.alloc(i64, count);
    var idx: usize = 0;
    for (0..out_tensor.shape.rank()) |i| {
        if (!reduced_axes[i]) continue;
        axes[idx] = @intCast(i);
        idx += 1;
    }
    return axes;
}

/// Compute gather output shape following StableHLO gather semantics.
///
/// Output dims = [batch dims from indices (excluding index_vector_dim)]
///             ++ [non-collapsed slice dims from operand].
///
/// Validates index_vector_dim, start_index_map length, collapsed_slice_dims
///  consistency (must have slice_size==1), and rank bounds.
fn compute_gather_output_dims(
    operand_dims: []const i64,
    indices_dims: []const i64,
    params: pr.GatherParams,
    out_buf: *[max_rank]i64,
) ?[]const i64 {
    if (operand_dims.len > max_rank) return null;
    if (params.slice_sizes.len != operand_dims.len) return null;
    if (params.index_vector_dim < 0) return null;
    const index_vector_dim: usize = @intCast(params.index_vector_dim);
    if (index_vector_dim > indices_dims.len) return null;
    // index_vector_dim is the axis in indices that holds the index coordinates
    // if it equals indices rank, each index is implicitly a scalar (length 1)
    const index_vector_len: usize = if (index_vector_dim == indices_dims.len)
        1
    else
        @intCast(indices_dims[index_vector_dim]);
    if (params.start_index_map.len != index_vector_len) return null;
    // collapsed dims must have slice_size==1 (they are removed from the output)
    var collapsed = @as([max_rank]bool, @splat(false));
    for (params.collapsed_slice_dims) |axis| {
        if (axis < 0) return null;
        const idx: usize = @intCast(axis);
        if (idx >= operand_dims.len) return null;
        if (collapsed[idx]) return null;
        collapsed[idx] = true;
        if (params.slice_sizes[idx] != 1) return null;
    }
    // Output = batch dims from indices (all except index_vector_dim)
    //        + offset dims from operand (non-collapsed slice dims)
    const out_rank = indices_dims.len - (if (index_vector_dim == indices_dims.len) @as(usize, 0) else @as(usize, 1)) +
        (operand_dims.len - params.collapsed_slice_dims.len);
    if (out_rank > max_rank) return null;
    // 1. batch dims from indices (skip the index_vector_dim axis)
    var out_i: usize = 0;
    for (indices_dims, 0..) |d, i| {
        if (i == index_vector_dim) continue;
        out_buf[out_i] = d;
        out_i += 1;
    }
    // 2. non-collapsed slice sizes from the operand
    for (0..operand_dims.len) |i| {
        if (collapsed[i]) continue;
        out_buf[out_i] = params.slice_sizes[i];
        out_i += 1;
    }
    return out_buf[0..out_i];
}

fn slice_matches(in_dims: []const i64, out_dims: []const i64, params: pr.SliceParams) bool {
    if (params.start_indices.len != in_dims.len) return false;
    if (params.limit_indices.len != in_dims.len) return false;
    if (params.strides.len != in_dims.len) return false;
    if (out_dims.len != in_dims.len) return false;
    for (in_dims, 0..) |dim, i| {
        const start = params.start_indices[i];
        const limit = params.limit_indices[i];
        const stride = params.strides[i];
        if (start < 0 or limit < 0 or stride <= 0) return false;
        if (dim < 0) return false;
        if (limit > dim or start >= limit) return false;
        const span: i64 = limit - start;
        const out: i64 = @divTrunc(span + stride - 1, stride);
        if (out_dims[i] != out) return false;
    }
    return true;
}

fn compute_slice_output_dims(allocator: std.mem.Allocator, in_dims: []const i64, params: pr.SliceParams) pr.BuildError![]const i64 {
    if (params.start_indices.len != in_dims.len) return error.SliceTypeMismatch;
    if (params.limit_indices.len != in_dims.len) return error.SliceTypeMismatch;
    if (params.strides.len != in_dims.len) return error.SliceTypeMismatch;
    const out_dims = try allocator.alloc(i64, in_dims.len);
    for (in_dims, 0..) |dim, i| {
        const start = params.start_indices[i];
        const limit = params.limit_indices[i];
        const stride = params.strides[i];
        if (start < 0 or limit < 0 or stride <= 0) return error.SliceTypeMismatch;
        if (dim < 0) return error.SliceTypeMismatch;
        if (limit > dim or start >= limit) return error.SliceTypeMismatch;
        const span: i64 = limit - start;
        out_dims[i] = @divTrunc(span + stride - 1, stride);
    }
    return out_dims;
}

fn concat_matches_op(op: *const pr.Op, out: Tensor, axis: i64) bool {
    if (axis < 0) return false;
    const axis_u: usize = @intCast(axis);
    if (out.shape.rank() == 0 or axis_u >= out.shape.rank()) return false;

    var out_sum: i64 = 0;
    for (op.inputs) |operand| {
        const t = operand.value.as_tensor();
        if (t.dtype != out.dtype) return false;
        if (t.shape.rank() != out.shape.rank()) return false;
        for (t.shape.dims, 0..) |d, i| {
            if (i == axis_u) continue;
            if (d != out.shape.dims[i]) return false;
        }
        out_sum += t.shape.dims[axis_u];
    }
    return out_sum == out.shape.dims[axis_u];
}

fn compute_concat_output_dims(alloc: std.mem.Allocator, inputs: []const *pr.Var, axis: i64) pr.BuildError![]const i64 {
    if (axis < 0) return error.ConcatTypeMismatch;
    const axis_u: usize = @intCast(axis);
    const first = inputs[0].as_tensor();
    if (first.shape.rank() == 0 or axis_u >= first.shape.rank()) return error.ConcatTypeMismatch;

    var out_dims = try alloc.dupe(i64, first.shape.dims);
    var total = out_dims[axis_u];
    for (inputs[1..]) |v| {
        const t = v.as_tensor();
        if (t.dtype != first.dtype) return error.ConcatTypeMismatch;
        if (t.shape.rank() != first.shape.rank()) return error.ConcatTypeMismatch;
        for (t.shape.dims, 0..) |d, i| {
            if (i == axis_u) continue;
            if (d != first.shape.dims[i]) return error.ConcatTypeMismatch;
        }
        total += t.shape.dims[axis_u];
    }
    out_dims[axis_u] = total;
    return out_dims;
}

fn compare_type_for_dtype(dt: pr.DType) pr.CompareType {
    return switch (dt) {
        .f16, .bf16, .f32, .f64 => .FLOAT,
        .i8, .i32, .i64 => .SIGNED,
        .u8, .u32, .u64, .bool => .UNSIGNED,
    };
}

/// Derive scatter params that invert a gather (for VJP scatter-add).
fn scatter_params_for_gather(params: pr.GatherParams) pr.ScatterParams {
    return .{
        .update_window_dims = params.offset_dims,
        .inserted_window_dims = params.collapsed_slice_dims,
        .scatter_dims_to_operand_dims = params.start_index_map,
        .index_vector_dim = params.index_vector_dim,
        .reduction = .add,
    };
}
