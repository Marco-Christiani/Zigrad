//! PR JSON graph emitter.
//!
//! Parameters and operations become nodes. Consumed variables become data-flow
//!  edges carrying value identity and type. Nodes also record every result so
//!  terminal and unused values retain their type information.
const std = @import("std");
const pr = @import("pr.zig");
const ops = @import("ops/ops.zig");
const zxpr = @import("zxpr.zig");

const Writer = std.Io.Writer;

/// Emit one function as a JSON graph.
///
/// Parameters and operations are nodes. Each operand use creates an edge from
///  its defining node.
pub fn emit(func: pr.Function, writer: *Writer) !void {
    return try emit_function(func, null, writer);
}

fn emit_function(func: pr.Function, function_id: ?pr.FunctionId, writer: *Writer) !void {
    try writer.writeAll("{");

    if (function_id) |id| {
        try writer.print("\"id\":{d},", .{@intFromEnum(id)});
    }
    try writer.writeAll("\"name\":");
    try write_json_string(writer, func.name);
    try writer.writeAll(",\"annotations\":");
    try emit_annotations(writer, func.annotations);

    try writer.writeAll(",\"nodes\":[");
    var node_idx: usize = 0;

    for (func.params, 0..) |param_var, pi| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_param_node(writer, param_var, pi);
        node_idx += 1;
    }

    for (func.ops, 0..) |op, oi| {
        if (node_idx > 0) try writer.writeAll(",");
        try emit_op_node(writer, op, oi);
        node_idx += 1;
    }
    try writer.writeAll("]");

    // A var whose producer cannot be named yields no edge, so the separator is
    //  written only once an edge is known to be emitted and the array holds no
    //  holes.
    try writer.writeAll(",\"edges\":[");
    var edge_idx: usize = 0;
    for (func.ops, 0..) |op, oi| {
        for (op.inputs, 0..) |operand, port| {
            const producer = producer_of(func, operand.value) orelse continue;
            if (edge_idx > 0) try writer.writeAll(",");
            try emit_var_edge(writer, operand.value, producer, oi, port);
            edge_idx += 1;
        }
    }
    try writer.writeAll("]");

    try writer.writeAll(",\"regions\":[");
    for (func.regions, 0..) |region, ri| {
        if (ri > 0) try writer.writeAll(",");
        try emit_region(writer, func, region);
    }
    try writer.writeAll("]");

    try writer.writeAll(",\"returns\":[");
    for (func.returns, 0..) |ret_var, i| {
        if (i > 0) try writer.writeAll(",");
        try write_value_id(writer, ret_var.id);
    }
    try writer.writeAll("]");

    try writer.writeAll("}\n");
}

/// Emit a Program as a JSON array of function graphs.
pub fn emit_program(program: *const pr.Program, writer: *Writer) !void {
    try writer.writeAll("[");
    for (program.functions(), program.function_ids(), 0..) |func, function_id, i| {
        if (i > 0) try writer.writeAll(",");
        try emit_function(func, function_id, writer);
    }
    try writer.writeAll("]\n");
}

/// Write a parameter or operation node id.
fn op_node_id(writer: *Writer, prefix: []const u8, index: usize) !void {
    try writer.writeAll("\"");
    try writer.writeAll(prefix);
    try writer.print("{d}", .{index});
    try writer.writeAll("\"");
}

fn emit_param_node(writer: *Writer, v: *const pr.Var, param_index: usize) !void {
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "p", param_index);
    try writer.writeAll(",\"kind\":\"param\"");
    try writer.writeAll(",\"label\":");
    try write_value_id(writer, v.id);
    try emit_outputs(writer, &.{v});
    try emit_zxpr_field_param(writer, v);
    try writer.writeAll("}");
}

fn emit_op_node(writer: *Writer, op: *const pr.Op, op_index: usize) !void {
    const prim = op.prim();
    try writer.writeAll("{\"id\":");
    try op_node_id(writer, "e", op_index);
    try writer.writeAll(",\"kind\":");
    try write_json_string(writer, @tagName(prim));
    try emit_param_attrs(writer, op);
    if (ops.has_local_jvp(prim)) {
        try writer.writeAll(",\"jvp\":true");
    }
    if (ops.has_local_vjp(prim)) {
        try writer.writeAll(",\"vjp\":true");
    }
    if (op.outputs.len > 0) {
        try writer.writeAll(",\"label\":\"");
        for (op.outputs, 0..) |out, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("%{d}", .{out.id});
        }
        try writer.writeAll(" = ");
        try writer.writeAll(@tagName(prim));
        try writer.writeAll("\"");
    }
    try emit_outputs(writer, op.outputs);
    try emit_zxpr_field_op(writer, op);
    try writer.writeAll("}");
}

/// Emit every value defined by one node.
///
/// Terminal and unused results have no outgoing edge, so their type information
///  must remain on the defining node.
fn emit_outputs(writer: *Writer, outputs: []const *const pr.Var) !void {
    try writer.writeAll(",\"outputs\":[");
    for (outputs, 0..) |out, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.writeAll("{\"var\":");
        try write_value_id(writer, out.id);
        try emit_aval_fields(writer, out.aval);
        try writer.writeAll("}");
    }
    try writer.writeAll("]");
}

/// Identify the node that defines a variable.
///
/// The producing op comes from `Var.defining_op`. Parameters are located by id
///  among the function's parameters.
const Producer = struct { id_prefix: []const u8, index: usize };

fn producer_of(func: pr.Function, v: *const pr.Var) ?Producer {
    if (v.defining_op) |op| {
        // Op ids are usually dense positions that index `func.ops` directly.
        //  The scan handles functions whose ids are not dense.
        if (op.id < func.ops.len and func.ops[op.id].id == op.id)
            return .{ .id_prefix = "e", .index = op.id };
        return .{ .id_prefix = "e", .index = func.op_index_by_id(op.id) orelse return null };
    }
    for (func.params, 0..) |param_var, pi|
        if (param_var.id == v.id) return .{ .id_prefix = "p", .index = pi };
    return null;
}

fn emit_var_edge(writer: *Writer, v: *const pr.Var, producer: Producer, target_op_idx: usize, port: usize) !void {
    try writer.writeAll("{\"source\":");
    try op_node_id(writer, producer.id_prefix, producer.index);
    try writer.writeAll(",\"target\":");
    try op_node_id(writer, "e", target_op_idx);
    try writer.print(",\"port\":{d}", .{port});

    try writer.writeAll(",\"var\":");
    try write_value_id(writer, v.id);

    try emit_aval_fields(writer, v.aval);

    try writer.writeAll("}");
}

fn emit_aval_fields(writer: *Writer, aval: pr.Aval) !void {
    switch (aval) {
        .tensor => |t| {
            try writer.writeAll(",\"dtype\":");
            try write_json_string(writer, @tagName(t.dtype));
            try writer.writeAll(",\"shape\":[");
            for (t.shape.dims, 0..) |d, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.print("{d}", .{d});
            }
            try writer.writeAll("]");
        },
    }
}

fn write_value_id(writer: *Writer, id: u32) !void {
    try writer.print("\"%{d}\"", .{id});
}

fn emit_zxpr_field_param(writer: *Writer, v: *const pr.Var) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_param_line(v, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

fn emit_zxpr_field_op(writer: *Writer, op: *const pr.Op) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_op_line(op, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

fn emit_zxpr_field_region(writer: *Writer, func: pr.Function, region: pr.Region) !void {
    var buf: [512]u8 = undefined;
    var zw: Writer = .fixed(&buf);
    zxpr.emit_region_block(func, region, &zw) catch return;
    const text = zw.buffered();
    if (text.len > 0) {
        try writer.writeAll(",\"zxpr\":");
        try write_json_string(writer, text);
    }
}

fn emit_param_attrs(writer: *Writer, op: *const pr.Op) !void {
    var has_attr = false;
    // Keep this switch exhaustive so every new Params variant defines its JSON form.
    switch (op.params) {
        .add, .subtract, .multiply, .divide, .maximum => {},
        .exp, .log, .rsqrt, .logistic => {},
        .select, .dot, .mm, .bmm => {},

        .literal => |lit| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"value\":");
            try emit_literal_json(writer, lit);
        },
        .reshape => |rp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_shape\":");
            try emit_i64_array(writer, rp.out_shape);
        },
        .broadcast_in_dim => |bp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_shape\":");
            try emit_i64_array(writer, bp.out_shape);
            try writer.writeAll(",\"broadcast_dims\":");
            try emit_i64_array(writer, bp.dimensions);
        },
        .transpose => |tp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"permutation\":");
            try emit_i64_array(writer, tp.permutation);
        },
        .reduce => |rp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"reduction\":");
            try write_json_string(writer, @tagName(rp.operation));
            try writer.writeByte(',');
            try writer.writeAll("\"reduce_axes\":");
            try emit_i64_array(writer, rp.axes);
        },
        .iota => |ip| {
            try open_attrs(writer, &has_attr);
            try writer.print("\"iota_dim\":{d}", .{ip.dimension});
            try writer.writeAll(",\"out_shape\":");
            try emit_i64_array(writer, ip.out_shape);
            try writer.writeAll(",\"out_dtype\":");
            try write_json_string(writer, @tagName(ip.out_dtype));
        },
        .convert => |dt| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"out_dtype\":");
            try write_json_string(writer, @tagName(dt));
        },
        .compare => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"direction\":");
            try write_json_string(writer, @tagName(cp.direction));
            try writer.writeAll(",\"compare_type\":");
            try write_json_string(writer, @tagName(cp.compare_type));
        },
        .dot_general => |dg| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"lhs_contracting\":");
            try emit_i64_array(writer, dg.lhs_contracting_dims);
            try writer.writeAll(",\"rhs_contracting\":");
            try emit_i64_array(writer, dg.rhs_contracting_dims);
            try writer.writeAll(",\"lhs_batch\":");
            try emit_i64_array(writer, dg.lhs_batch_dims);
            try writer.writeAll(",\"rhs_batch\":");
            try emit_i64_array(writer, dg.rhs_batch_dims);
        },
        .convolution => |conv| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"window_strides\":");
            try emit_i64_array(writer, conv.window_strides);
            try writer.writeAll(",\"padding\":");
            try emit_i64_array(writer, conv.padding);
            try writer.writeAll(",\"lhs_dilation\":");
            try emit_i64_array(writer, conv.lhs_dilation);
            try writer.writeAll(",\"rhs_dilation\":");
            try emit_i64_array(writer, conv.rhs_dilation);
            try writer.writeAll(",\"window_reversal\":");
            try emit_bool_array(writer, conv.window_reversal);
            try writer.writeAll(",\"input_spatial_dimensions\":");
            try emit_i64_array(writer, conv.dimensions.input_spatial_dimensions);
            try writer.writeAll(",\"kernel_spatial_dimensions\":");
            try emit_i64_array(writer, conv.dimensions.kernel_spatial_dimensions);
            try writer.writeAll(",\"output_spatial_dimensions\":");
            try emit_i64_array(writer, conv.dimensions.output_spatial_dimensions);
            try writer.print(",\"input_batch_dimension\":{d},\"input_feature_dimension\":{d}", .{
                conv.dimensions.input_batch_dimension,
                conv.dimensions.input_feature_dimension,
            });
            try writer.print(",\"kernel_input_feature_dimension\":{d},\"kernel_output_feature_dimension\":{d}", .{
                conv.dimensions.kernel_input_feature_dimension,
                conv.dimensions.kernel_output_feature_dimension,
            });
            try writer.print(",\"output_batch_dimension\":{d},\"output_feature_dimension\":{d}", .{
                conv.dimensions.output_batch_dimension,
                conv.dimensions.output_feature_dimension,
            });
            try writer.print(",\"feature_group_count\":{d},\"batch_group_count\":{d}", .{
                conv.feature_group_count,
                conv.batch_group_count,
            });
        },
        .gather => |g| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"slice_sizes\":");
            try emit_i64_array(writer, g.slice_sizes);
            try writer.writeAll(",\"offset_dims\":");
            try emit_i64_array(writer, g.offset_dims);
            try writer.writeAll(",\"collapsed_slice_dims\":");
            try emit_i64_array(writer, g.collapsed_slice_dims);
            try writer.writeAll(",\"start_index_map\":");
            try emit_i64_array(writer, g.start_index_map);
            try writer.print(",\"index_vector_dim\":{d}", .{g.index_vector_dim});
        },
        .scatter => |s| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"update_window_dims\":");
            try emit_i64_array(writer, s.update_window_dims);
            try writer.writeAll(",\"inserted_window_dims\":");
            try emit_i64_array(writer, s.inserted_window_dims);
            try writer.writeAll(",\"scatter_dims_to_operand_dims\":");
            try emit_i64_array(writer, s.scatter_dims_to_operand_dims);
            try writer.print(",\"index_vector_dim\":{d}", .{s.index_vector_dim});
            try writer.writeAll(",\"reduction\":");
            try write_json_string(writer, @tagName(s.reduction));
        },
        .slice => |s| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"start\":");
            try emit_i64_array(writer, s.start_indices);
            try writer.writeAll(",\"limit\":");
            try emit_i64_array(writer, s.limit_indices);
            try writer.writeAll(",\"strides\":");
            try emit_i64_array(writer, s.strides);
        },
        .concatenate => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.print("\"axis\":{d}", .{cp.axis});
        },
        .call => |cp| {
            try open_attrs(writer, &has_attr);
            try writer.print("\"callee\":{d}", .{@intFromEnum(cp.callee)});
        },
        .custom_call => |cc| {
            try open_attrs(writer, &has_attr);
            try writer.writeAll("\"target\":");
            try write_json_string(writer, cc.target_name);
            if (cc.payload.len > 0) {
                try writer.writeAll(",\"payload\":");
                try emit_bytes(writer, cc.payload);
            }
            if (cc.has_side_effect) {
                try writer.writeAll(",\"side_effect\":true");
            }
        },
    }
    if (has_attr) try writer.writeAll("}");
}

fn open_attrs(writer: *Writer, has_attr: *bool) !void {
    if (!has_attr.*) {
        try writer.writeAll(",\"attrs\":{");
        has_attr.* = true;
    } else {
        try writer.writeAll(",");
    }
}

fn emit_bool_array(writer: *Writer, values: []const bool) !void {
    try writer.writeAll("[");
    for (values, 0..) |value, i| {
        if (i != 0) try writer.writeAll(",");
        try writer.writeAll(if (value) "true" else "false");
    }
    try writer.writeAll("]");
}

fn emit_region(writer: *Writer, func: pr.Function, region: pr.Region) !void {
    try writer.writeAll("{\"name\":");
    try write_json_string(writer, region.name);

    try writer.writeAll(",\"annotations\":");
    try emit_annotations(writer, region.annotations);

    try writer.writeAll(",\"node_ids\":[");
    var first = true;
    for (region.op_ids) |op_id| {
        if (func.op_by_id(op_id) == null) continue;
        if (!first) try writer.writeAll(",");
        try writer.print("\"e{d}\"", .{op_id});
        first = false;
    }
    try writer.writeAll("]");

    try emit_zxpr_field_region(writer, func, region);
    try writer.writeAll("}");
}

fn emit_annotations(writer: *Writer, annotations: []const pr.Annotation) !void {
    try writer.writeAll("{");
    for (annotations, 0..) |annotation, index| {
        if (index > 0) try writer.writeAll(",");
        try write_json_string(writer, annotation.name);
        try writer.writeAll(":");
        try emit_annotation_value(writer, annotation.value);
    }
    try writer.writeAll("}");
}

fn emit_annotation_value(writer: *Writer, value: pr.AnnotationValue) !void {
    switch (value) {
        .unit => try writer.writeAll("null"),
        .boolean => |item| try writer.writeAll(if (item) "true" else "false"),
        .integer => |item| try writer.print("{d}", .{item}),
        .floating_point => |item| {
            if (std.math.isFinite(item)) {
                try writer.print("{d}", .{item});
            } else if (std.math.isNan(item)) {
                try write_json_string(writer, "nan");
            } else if (item < 0) {
                try write_json_string(writer, "-inf");
            } else {
                try write_json_string(writer, "inf");
            }
        },
        .string => |item| try write_json_string(writer, item),
        .strings => |items| {
            try writer.writeAll("[");
            for (items, 0..) |item, index| {
                if (index > 0) try writer.writeAll(",");
                try write_json_string(writer, item);
            }
            try writer.writeAll("]");
        },
        .bytes => |items| try emit_bytes(writer, items),
    }
}

fn emit_bytes(writer: *Writer, items: []const u8) !void {
    try writer.writeAll("[");
    for (items, 0..) |item, index| {
        if (index > 0) try writer.writeAll(",");
        try writer.print("{d}", .{item});
    }
    try writer.writeAll("]");
}

fn write_json_string(writer: *Writer, s: []const u8) !void {
    try writer.writeAll("\"");
    var start: usize = 0;
    for (s, 0..) |c, i| {
        switch (c) {
            '"', '\\', 0x00...0x1f => {
                if (i > start) try writer.writeAll(s[start..i]);
                switch (c) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => try writer.print("\\u{x:0>4}", .{@as(u16, c)}),
                }
                start = i + 1;
            },
            else => {},
        }
    }
    if (start < s.len) try writer.writeAll(s[start..]);
    try writer.writeAll("\"");
}

fn emit_i64_array(writer: *Writer, items: []const i64) !void {
    try writer.writeAll("[");
    for (items, 0..) |v, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.print("{d}", .{v});
    }
    try writer.writeAll("]");
}

/// Write one PR literal as JSON.
///
/// BF16 values widen to F32. Non-finite floats use the strings `"inf"`, `"-inf"`,
///  and `"nan"` because JSON has no numeric representation for them.
fn emit_literal_json(writer: *Writer, lit: pr.Literal) !void {
    switch (lit) {
        .f16 => |v| try emit_float_json(writer, pr.DType.f16.decode(f32, v)),
        .bf16 => |v| try emit_float_json(writer, pr.DType.bf16.decode(f32, v)),
        .bool => |v| try writer.writeAll(if (v) "true" else "false"),
        inline .f32, .f64 => |v| try emit_float_json(writer, v),
        inline .i8, .u8, .i32, .i64, .u32, .u64 => |v| try writer.print("{d}", .{v}),
    }
}

fn emit_float_json(writer: *Writer, v: anytype) !void {
    if (std.math.isNan(v)) return try writer.writeAll("\"nan\"");
    if (std.math.isPositiveInf(v)) return try writer.writeAll("\"inf\"");
    if (std.math.isNegativeInf(v)) return try writer.writeAll("\"-inf\"");
    try writer.print("{d}", .{v});
}

test emit {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.mm(a, c);
    const func = try b.finish(.{ .returns = &.{d} });

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    try std.testing.expect(result.len > 0);
    try std.testing.expect(result[0] == '{');
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"main\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"nodes\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"edges\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"returns\":[\"%2\"]") != null);

    // Nodes are ops: params "p0","p1", op "e0"
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"p0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"p1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"id\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"param\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"mm\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"jvp\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"vjp\":true") != null);

    // Param nodes have value-id labels
    try std.testing.expect(std.mem.indexOf(u8, result, "\"label\":\"%0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"label\":\"%1\"") != null);

    // Edges are vars: carry value id, dtype, shape
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"p0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"p1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"target\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"%0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"%1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"shape\":[2,3]") != null);

    // ZXPR snippets on nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"zxpr\":\"%0: 2x3<f32>\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "mm[M=2, K=3, N=2") != null);
}

test "json with regions" {
    const kernel = @import("kernel");
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "k");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 2 });
    const c = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", &.{kernel.provider_annotation("tvm")});
    const add1 = try b.add(a, c);
    const add2 = try b.add(add1, c);
    try b.pop_region();

    _ = try b.add(add2, c);
    var func = try b.finish(.{ .returns = &.{add2} });
    func.annotations = &.{.{ .name = "example.function", .value = .{ .boolean = true } }};

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    try std.testing.expect(std.mem.indexOf(u8, result, "\"zigrad.kernel.provider\":\"tvm\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"example.function\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"tvm-kernel\"") != null);
    // Region node_ids reference op nodes (e0, e1)
    try std.testing.expect(std.mem.indexOf(u8, result, "\"node_ids\":[\"e0\",\"e1\"]") != null);
}

test "json with reshape and transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "shapes");
    defer b.deinit();

    const x = try b.param_tensor(.f64, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const func = try b.finish(.{ .returns = &.{r} });

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    // Op nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"transpose\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"permutation\":[1,0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"reshape\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"out_shape\":[6]") != null);
    // Var edges carry type info
    try std.testing.expect(std.mem.indexOf(u8, result, "\"var\":\"%0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"dtype\":\"f64\"") != null);
}

test "json with broadcast and literal" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "bcast");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const broadcasted = try b.broadcast_in_dim(one, &.{ 2, 3 }, &.{});
    const func = try b.finish(.{ .returns = &.{broadcasted} });

    var buf: [2048]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();

    // Op nodes
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"literal\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"kind\":\"broadcast_in_dim\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"broadcast_dims\":[]") != null);
    // The literal is e0 and the broadcast is e1. Their edge carries value %0.
    try std.testing.expect(std.mem.indexOf(u8, result, "\"source\":\"e0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"target\":\"e1\"") != null);
}

test "json output is valid JSON" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "matmul_add");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const w = try b.param_tensor(.f32, &.{ 3, 4 });
    const bias = try b.param_tensor(.f32, &.{4});
    const prod = try b.mm(a, w);
    const bias_bc = try b.broadcast_in_dim(bias, &.{ 2, 4 }, &.{1});
    const out = try b.add(prod, bias_bc);
    const func = try b.finish(.{ .returns = &.{out} });

    var buf: [8192]u8 = undefined;
    var wr: Writer = .fixed(&buf);
    try emit(func, &wr);

    const result = wr.buffered();

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, result, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    try std.testing.expectEqualStrings("matmul_add", root.get("name").?.string);

    // 3 params + 3 ops = 6 nodes
    const nodes = root.get("nodes").?.array;
    try std.testing.expectEqual(6, nodes.items.len);

    // Edges = vars flowing between ops
    const edges = root.get("edges").?.array;
    try std.testing.expect(edges.items.len > 0);

    // Each edge has var name, dtype, shape
    const first_edge = edges.items[0].object;
    try std.testing.expect(first_edge.get("var") != null);
    try std.testing.expect(first_edge.get("dtype") != null);
    try std.testing.expect(first_edge.get("shape") != null);

    const returns = root.get("returns").?.array;
    try std.testing.expectEqual(1, returns.items.len);
}

test "every op input yields an edge" {
    // Counting the edges rather than pattern-matching the text catches the
    //  whole class: a dropped edge is a dropped data-flow fact whether or not
    //  the document still parses.
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "flow");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 2 });
    const y = try b.param_tensor(.f32, &.{ 2, 2 });
    var acc = try b.add(x, y);
    for (0..8) |_| acc = try b.multiply(acc, y);
    const func = try b.finish(.{ .returns = &.{acc} });

    var expected: usize = 0;
    for (func.ops) |op| expected += op.inputs.len;

    var buf: [16384]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, w.buffered(), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(expected, parsed.value.object.get("edges").?.array.items.len);
}

test "a value no op consumes still carries its type" {
    // The returned product is consumed by nothing, so no edge describes it. A
    //  reader reconstructing the function needs its shape as much as any
    //  operand's.
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "terminal");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 3, 4 });
    const out = try b.mm(x, y);
    const func = try b.finish(.{ .returns = &.{out} });

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, w.buffered(), .{});
    defer parsed.deinit();

    const returned = parsed.value.object.get("returns").?.array.items[0].string;
    var found: ?std.json.Value = null;
    for (parsed.value.object.get("nodes").?.array.items) |node| {
        for (node.object.get("outputs").?.array.items) |o| {
            if (std.mem.eql(u8, o.object.get("var").?.string, returned)) found = o;
        }
    }
    const shape = found.?.object.get("shape").?.array;
    try std.testing.expectEqual(2, shape.items.len);
    try std.testing.expectEqual(2, shape.items[0].integer);
    try std.testing.expectEqual(4, shape.items[1].integer);
}

test "non-finite literals stay parseable" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "masked");
    defer b.deinit();
    const neg_inf = try b.literal_scalar(.{ .f32 = -std.math.inf(f32) });
    const out = try b.broadcast_in_dim(neg_inf, &.{ 2, 2 }, &.{});
    const func = try b.finish(.{ .returns = &.{out} });

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit(func, &w);

    const result = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, result, "\"value\":\"-inf\"") != null);

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, result, .{});
    defer parsed.deinit();
}

test emit_program {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b1 = try pr.FunctionBuilder.init(&program, "f1");
    defer b1.deinit();
    const x = try b1.param_tensor(.f32, &.{4});
    const func1 = try b1.finish(.{ .returns = &.{x} });
    _ = try program.add_function(func1);

    var b2 = try pr.FunctionBuilder.init(&program, "f2");
    defer b2.deinit();
    const y = try b2.param_tensor(.f32, &.{8});
    const func2 = try b2.finish(.{ .returns = &.{y} });
    _ = try program.add_function(func2);

    var buf: [4096]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try emit_program(&program, &w);

    const result = w.buffered();

    try std.testing.expect(result[0] == '[');
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"f1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"name\":\"f2\"") != null);
}

test "emit_program keeps the array schema for one function" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const x = try builder.param_tensor(.f32, &.{1});
    _ = try program.add_function(try builder.finish(.{ .returns = &.{x} }));

    var buf: [2048]u8 = undefined;
    var writer: Writer = .fixed(&buf);
    try emit_program(&program, &writer);

    const result = writer.buffered();
    try std.testing.expect(std.mem.startsWith(u8, result, "["));
    try std.testing.expect(std.mem.endsWith(u8, result, "]\n"));
}

test "emit_program identifies monotonic call targets" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const checkpoint = program.checkpoint_appends();
    var removed = try pr.FunctionBuilder.init(&program, "removed");
    defer removed.deinit();
    _ = try program.add_function(try removed.finish(.{ .returns = &.{} }));
    program.restore_appends(checkpoint);

    var callee = try pr.FunctionBuilder.init(&program, "callee");
    defer callee.deinit();
    const callee_input = try callee.param_tensor(.f32, &.{});
    const callee_id = try program.add_function(try callee.finish(.{ .returns = &.{callee_input} }));

    var caller = try pr.FunctionBuilder.init(&program, "caller");
    defer caller.deinit();
    const caller_input = try caller.param_tensor(.f32, &.{});
    const call_op = try caller.call(callee_id, &.{caller_input});
    _ = try program.add_function(try caller.finish(.{ .returns = call_op.outputs }));

    var output: Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try emit_program(&program, &output.writer);

    const json = output.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\":1,\"name\":\"callee\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"callee\":1") != null);
}
