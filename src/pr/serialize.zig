//! Versioned binary serialization for PR programs.
//!
//! Records value definitions with their types, references other values by id,
//!  and reconstructs derived def-use links while parsing.
//!
//! `emit` and `parse` cover whole programs. Public codec primitives expose the
//!  reflected encoding for individual fragments. Function definitions, calls,
//!  and entry selection retain their program-local identities.
//!
//! Separate schema hashes identify whole programs, value fragments, and
//!  operation fragments.

const std = @import("std");
const pr = @import("pr.zig");

const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;

/// Eight-byte marker at the start of every PR wire document.
pub const magic = "ZGPRWIRE";
/// Current PR wire format version.
pub const version: u32 = 9;
const header_size = magic.len + @sizeOf(u32) + @sizeOf(u64);

// This count forces a wire-version decision when `Prim` or `Params` changes.
const wire_prim_count = 29;

comptime {
    const prim_fields = std.meta.fields(pr.Prim);
    const params_fields = @typeInfo(pr.Params).@"union".fields;
    if (prim_fields.len != wire_prim_count or params_fields.len != wire_prim_count)
        @compileError("Prim changed: update the PR wire version and wire_prim_count");
    for (prim_fields, params_fields) |prim_field, params_field| {
        if (!std.mem.eql(u8, prim_field.name, params_field.name))
            @compileError("Prim and Params variants must have matching order and names");
    }
}

/// Fingerprint of the complete PR wire layout.
pub const schema_hash = compute_program_schema_hash();
/// Fingerprint of a value fragment encoded as dtype followed by dimensions.
pub const value_schema_hash = compute_value_schema_hash();
/// Fingerprint of an operation fragment encoded as id followed by parameters.
pub const operation_schema_hash = compute_operation_schema_hash();

/// Errors produced while emitting a PR wire document.
pub const EmitError = Writer.Error || error{
    LengthOverflow,
    UnknownFunctionId,
};

const DecodeError = Allocator.Error || error{
    Truncated,
    LengthOverflow,
    InvalidBool,
    InvalidEnumTag,
    InvalidUnionTag,
    InvalidParamsTag,
    DuplicateVarId,
    VarIdOutOfRange,
    UnknownVarId,
    DuplicateOpId,
    UnknownOpId,
    VarCountMismatch,
};

/// Errors produced while parsing or validating a PR wire document.
pub const ParseError = DecodeError || pr.ValidationError || pr.FunctionRegistrationError || error{
    InvalidMagic,
    UnsupportedVersion,
    SchemaMismatch,
    TrailingData,
};

/// Emit a complete PR program in the versioned binary wire format.
pub fn emit(program: *const pr.Program, writer: *Writer) EmitError!void {
    try writer.writeAll(magic);
    try writer.writeInt(u32, version, .little);
    try writer.writeInt(u64, schema_hash, .little);
    try write_length(writer, program.functions().len);
    if (program.entry) |entry|
        if (program.get_function_by_id(entry) == null) return error.UnknownFunctionId;
    try write_value(?pr.FunctionId, writer, program.entry);

    for (program.functions(), program.function_ids()) |func, id| {
        try write_value(pr.FunctionId, writer, id);
        try write_value([]const u8, writer, func.name);
        try write_value([]const pr.Annotation, writer, func.annotations);
        try writer.writeInt(u32, func.var_count, .little);

        try write_length(writer, func.params.len);
        for (func.params) |param| try write_var_definition(writer, param);

        try write_length(writer, func.ops.len);
        for (func.ops) |op| {
            try writer.writeInt(u32, op.id, .little);

            try write_length(writer, op.inputs.len);
            for (op.inputs) |operand|
                try writer.writeInt(u32, operand.value.id, .little);

            try write_length(writer, op.outputs.len);
            for (op.outputs) |output| try write_var_definition(writer, output);

            try write_program_params(program, writer, op.params);
        }

        try write_length(writer, func.regions.len);
        for (func.regions) |region| {
            try writer.writeInt(u32, region.id, .little);
            try write_value([]const u8, writer, region.name);
            try write_value([]const pr.Annotation, writer, region.annotations);
            try write_value([]const u32, writer, region.op_ids);
        }

        try write_length(writer, func.returns.len);
        for (func.returns) |return_var|
            try writer.writeInt(u32, return_var.id, .little);
    }
}

fn write_program_params(program: *const pr.Program, writer: *Writer, params: pr.Params) EmitError!void {
    if (params == .call and program.get_function_by_id(params.call.callee) == null)
        return error.UnknownFunctionId;
    try write_value(pr.Params, writer, params);
}

/// Fixed fields at the start of a PR wire payload.
pub const Header = struct {
    /// Raw format marker found in the payload.
    magic: [magic.len]u8,
    /// Raw wire-format version found in the payload.
    version: u32,
    /// Raw reflected schema fingerprint found in the payload.
    schema_hash: u64,

    /// Return whether this build can decode the payload.
    pub fn is_supported(self: Header) bool {
        self.validate() catch return false;
        return true;
    }

    pub fn format(self: Header, writer: *Writer) Writer.Error!void {
        try writer.print("magic={[magic]s} version={[version]d} schema_hash={[schema_hash]d}", self);
    }

    fn validate(self: Header) error{ InvalidMagic, UnsupportedVersion, SchemaMismatch }!void {
        if (!std.mem.eql(u8, magic, &self.magic)) return error.InvalidMagic;
        if (self.version != version) return error.UnsupportedVersion;
        if (self.schema_hash != schema_hash) return error.SchemaMismatch;
    }
};

/// Read a PR wire header without validating.
pub fn read_header(bytes: []const u8) error{Truncated}!Header {
    if (bytes.len < header_size) return error.Truncated;
    var header = Header{
        .magic = undefined,
        .version = std.mem.readInt(u32, bytes[magic.len..][0..@sizeOf(u32)], .little),
        .schema_hash = std.mem.readInt(u64, bytes[magic.len + @sizeOf(u32) ..][0..@sizeOf(u64)], .little),
    };
    @memcpy(&header.magic, bytes[0..magic.len]);
    return header;
}

/// Parse and validate a complete PR program.
///
/// The returned program copies parsed data into its arena, so the input bytes
///  may be released after return.
pub fn parse(backing_allocator: Allocator, bytes: []const u8) ParseError!pr.Program {
    const header = try read_header(bytes);
    try header.validate();
    var reader = Reader{ .bytes = bytes, .pos = header_size };

    var program = pr.Program.init(backing_allocator);
    errdefer program.deinit();
    const arena = program.allocator();

    const function_count = try reader.read_length();
    const entry = try read_value(?pr.FunctionId, &reader, arena);
    const functions = try arena.alloc(pr.Function, function_count);
    const function_ids = try arena.alloc(pr.FunctionId, function_count);
    for (functions, function_ids) |*func, *id| {
        id.* = try read_value(pr.FunctionId, &reader, arena);
        func.* = try read_function(&reader, arena);
    }
    for (functions, function_ids) |func, id|
        _ = try program.add_function_with_id(func, id);
    if (entry) |id|
        program.set_entry(id) catch return error.EntryFunctionMissing;

    if (reader.pos != bytes.len) return error.TrailingData;
    try pr.validate_program(&program);
    return program;
}

fn write_var_definition(writer: *Writer, variable: *const pr.Var) EmitError!void {
    try writer.writeInt(u32, variable.id, .little);
    const tensor = variable.aval.as_tensor();
    try write_value(pr.DType, writer, tensor.dtype);
    try write_value([]const i64, writer, tensor.shape.dims);
}

fn read_var_definition(
    reader: *Reader,
    arena: Allocator,
    vars: []?*pr.Var,
    defining_op: ?*const pr.Op,
) DecodeError!*pr.Var {
    const id = try reader.read_int(u32);
    if (id >= vars.len) return error.VarIdOutOfRange;
    if (vars[id] != null) return error.DuplicateVarId;

    const dtype = try read_value(pr.DType, reader, arena);
    const dims = try read_value([]const i64, reader, arena);
    const variable = try arena.create(pr.Var);
    variable.* = .{
        .id = id,
        .aval = .{ .tensor = .{
            .dtype = dtype,
            .shape = .{ .dims = dims },
        } },
        .defining_op = defining_op,
    };
    vars[id] = variable;
    return variable;
}

fn read_function(reader: *Reader, arena: Allocator) DecodeError!pr.Function {
    const name = try read_value([]const u8, reader, arena);
    const function_annotations = try read_value([]const pr.Annotation, reader, arena);
    const var_count = try reader.read_int(u32);
    const vars = try arena.alloc(?*pr.Var, var_count);
    @memset(vars, null);

    const param_count = try reader.read_length();
    const params = try arena.alloc(*pr.Var, param_count);
    for (params) |*param|
        param.* = try read_var_definition(reader, arena, vars, null);

    const op_count = try reader.read_length();
    const function_ops = try arena.alloc(*pr.Op, op_count);
    for (function_ops, 0..) |*op_slot, op_index| {
        const id = try reader.read_int(u32);
        for (function_ops[0..op_index]) |existing|
            if (existing.id == id) return error.DuplicateOpId;

        const input_count = try reader.read_length();
        const inputs = try arena.alloc(pr.Operand, input_count);
        for (inputs) |*operand| {
            const input_id = try reader.read_int(u32);
            operand.value = try lookup_var(vars, input_id);
        }

        const op = try arena.create(pr.Op);
        const output_count = try reader.read_length();
        const outputs = try arena.alloc(*pr.Var, output_count);
        for (outputs) |*output|
            output.* = try read_var_definition(reader, arena, vars, op);

        const params_payload = try read_params(reader, arena);
        op.* = .{
            .id = id,
            .inputs = inputs,
            .outputs = outputs,
            .params = params_payload,
        };
        op_slot.* = op;

        for (inputs, 0..) |*operand, index| {
            const value = operand.value;
            operand.attach(value, op, @intCast(index));
        }
    }

    const region_count = try reader.read_length();
    const regions = try arena.alloc(pr.Region, region_count);
    for (regions) |*region| {
        const id = try reader.read_int(u32);
        const region_name = try read_value([]const u8, reader, arena);
        const annotations = try read_value([]const pr.Annotation, reader, arena);
        const op_ids = try read_value([]const u32, reader, arena);
        for (op_ids) |op_id| {
            var found = false;
            for (function_ops) |op| {
                if (op.id == op_id) {
                    found = true;
                    break;
                }
            }
            if (!found) return error.UnknownOpId;
        }
        region.* = .{
            .id = id,
            .name = region_name,
            .annotations = annotations,
            .op_ids = op_ids,
        };
    }

    const return_count = try reader.read_length();
    const returns = try arena.alloc(*pr.Var, return_count);
    for (returns) |*return_var| {
        const id = try reader.read_int(u32);
        return_var.* = try lookup_var(vars, id);
    }

    for (vars) |variable|
        if (variable == null) return error.VarCountMismatch;

    return .{
        .name = name,
        .annotations = function_annotations,
        .params = params,
        .returns = returns,
        .ops = function_ops,
        .regions = regions,
        .var_count = var_count,
    };
}

fn lookup_var(vars: []const ?*pr.Var, id: u32) DecodeError!*pr.Var {
    if (id >= vars.len) return error.VarIdOutOfRange;
    return vars[id] orelse error.UnknownVarId;
}

/// Write a slice or container length as a `u32`.
pub fn write_length(writer: *Writer, length: usize) EmitError!void {
    try writer.writeInt(u32, std.math.cast(u32, length) orelse return error.LengthOverflow, .little);
}

/// Encode one value of any supported type by reflection over `T`.
pub fn write_value(comptime T: type, writer: *Writer, value: T) EmitError!void {
    switch (@typeInfo(T)) {
        .void => {},
        .bool => try writer.writeByte(@intFromBool(value)),
        .int => |info| {
            if (info.bits % 8 != 0)
                @compileError("PR wire integers must occupy whole bytes");
            try writer.writeInt(T, value, .little);
        },
        .float => |info| {
            const Bits = std.meta.Int(.unsigned, info.bits);
            try writer.writeInt(Bits, @bitCast(value), .little);
        },
        .@"enum" => try write_enum(T, writer, value),
        .optional => |info| {
            if (value) |payload| {
                try writer.writeByte(1);
                try write_value(info.child, writer, payload);
            } else {
                try writer.writeByte(0);
            }
        },
        .pointer => |info| {
            if (info.size != .slice)
                @compileError("PR wire supports slices but not single-item pointers");
            try write_length(writer, value.len);
            for (value) |item| try write_value(info.child, writer, item);
        },
        .@"struct" => |info| {
            inline for (info.fields) |field|
                try write_value(field.type, writer, @field(value, field.name));
        },
        .@"union" => |info| {
            const Tag = info.tag_type orelse
                @compileError("PR wire requires tagged unions");
            const tag = std.meta.activeTag(value);
            try write_enum(Tag, writer, tag);
            inline for (info.fields) |field| {
                if (tag == @field(Tag, field.name))
                    try write_value(field.type, writer, @field(value, field.name));
            }
        },
        else => @compileError("unsupported PR wire type: " ++ @typeName(T)),
    }
}

fn write_enum(comptime T: type, writer: *Writer, value: T) EmitError!void {
    try writer.writeInt(u32, @intCast(@intFromEnum(value)), .little);
}

fn read_params(reader: *Reader, arena: Allocator) DecodeError!pr.Params {
    const tag_value = try reader.read_int(u32);
    inline for (@typeInfo(pr.Params).@"union".fields) |field| {
        const tag = @field(pr.Prim, field.name);
        if (tag_value == @intFromEnum(tag)) {
            return @unionInit(
                pr.Params,
                field.name,
                try read_value(field.type, reader, arena),
            );
        }
    }
    return error.InvalidParamsTag;
}

/// Decode one value of any supported type, allocating slices from `arena`.
pub fn read_value(comptime T: type, reader: *Reader, arena: Allocator) DecodeError!T {
    return switch (@typeInfo(T)) {
        .void => {},
        .bool => switch (try reader.read_int(u8)) {
            0 => false,
            1 => true,
            else => error.InvalidBool,
        },
        .int => |info| blk: {
            if (info.bits % 8 != 0)
                @compileError("PR wire integers must occupy whole bytes");
            break :blk try reader.read_int(T);
        },
        .float => |info| blk: {
            const Bits = std.meta.Int(.unsigned, info.bits);
            break :blk @bitCast(try reader.read_int(Bits));
        },
        .@"enum" => try read_enum(T, reader),
        .optional => |info| switch (try reader.read_int(u8)) {
            0 => null,
            1 => try read_value(info.child, reader, arena),
            else => error.InvalidBool,
        },
        .pointer => |info| blk: {
            if (info.size != .slice)
                @compileError("PR wire supports slices but not single-item pointers");
            const length = try reader.read_length();
            const items = try arena.alloc(info.child, length);
            for (items) |*item| item.* = try read_value(info.child, reader, arena);
            break :blk items;
        },
        .@"struct" => |info| blk: {
            var result: T = undefined;
            inline for (info.fields) |field|
                @field(result, field.name) = try read_value(field.type, reader, arena);
            break :blk result;
        },
        .@"union" => |info| blk: {
            const Tag = info.tag_type orelse
                @compileError("PR wire requires tagged unions");
            const tag = try read_enum(Tag, reader);
            inline for (info.fields) |field| {
                if (tag == @field(Tag, field.name)) {
                    break :blk @unionInit(
                        T,
                        field.name,
                        try read_value(field.type, reader, arena),
                    );
                }
            }
            return error.InvalidUnionTag;
        },
        else => @compileError("unsupported PR wire type: " ++ @typeName(T)),
    };
}

fn read_enum(comptime T: type, reader: *Reader) DecodeError!T {
    const value = try reader.read_int(u32);
    return std.enums.fromInt(T, value) orelse error.InvalidEnumTag;
}

/// A cursor over encoded bytes.
pub const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn take(self: *Reader, length: usize) DecodeError![]const u8 {
        if (length > self.bytes.len -| self.pos) return error.Truncated;
        const result = self.bytes[self.pos..][0..length];
        self.pos += length;
        return result;
    }

    pub fn read_int(self: *Reader, comptime T: type) DecodeError!T {
        const bytes = try self.take(@sizeOf(T));
        return std.mem.readInt(T, bytes[0..@sizeOf(T)], .little);
    }

    pub fn read_length(self: *Reader) DecodeError!usize {
        return @intCast(try self.read_int(u32));
    }
};

fn compute_program_schema_hash() u64 {
    @setEvalBranchQuota(100_000);
    var hash: u64 = 14695981039346656037;
    hash_bytes(&hash, "Program:function_count,?entry_id,[Function:id,name,[Annotation],var_count,[param],[Op],[Region],[return]];" ++
        "param:id,dtype,dims;Op:id,[input id],[output],Params;" ++
        "Region:id,name,[Annotation],[op id];output:id,dtype,dims");
    hash_type(&hash, pr.DType);
    hash_type(&hash, pr.Params);
    hash_type(&hash, pr.Annotation);
    return hash;
}

fn compute_value_schema_hash() u64 {
    var hash: u64 = 14695981039346656037;
    hash_bytes(&hash, "value:dtype,dims");
    hash_type(&hash, pr.DType);
    hash_type(&hash, []const i64);
    return hash;
}

fn compute_operation_schema_hash() u64 {
    @setEvalBranchQuota(100_000);
    var hash: u64 = 14695981039346656037;
    hash_bytes(&hash, "operation:id,Params");
    hash_type(&hash, u32);
    hash_type(&hash, pr.Params);
    return hash;
}

test read_header {
    var bytes: [header_size]u8 = undefined;
    @memcpy(bytes[0..magic.len], magic);
    std.mem.writeInt(u32, bytes[magic.len..][0..@sizeOf(u32)], version, .little);
    std.mem.writeInt(u64, bytes[magic.len + @sizeOf(u32) ..][0..@sizeOf(u64)], schema_hash, .little);

    const header = try read_header(&bytes);
    try std.testing.expect(header.is_supported());
    try std.testing.expectError(error.Truncated, read_header(bytes[0 .. bytes.len - 1]));

    bytes[magic.len] +%= 1;
    try std.testing.expect(!(try read_header(&bytes)).is_supported());
}

fn hash_type(hash: *u64, comptime T: type) void {
    switch (@typeInfo(T)) {
        .void => hash_bytes(hash, "void"),
        .bool => hash_bytes(hash, "bool"),
        .int => |info| {
            hash_bytes(hash, if (info.signedness == .signed) "sint" else "uint");
            hash_int(hash, info.bits);
        },
        .float => |info| {
            hash_bytes(hash, "float");
            hash_int(hash, info.bits);
        },
        .@"enum" => |info| {
            hash_bytes(hash, "enum");
            hash_type(hash, info.tag_type);
            inline for (info.fields) |field| {
                hash_bytes(hash, field.name);
                hash_int(hash, field.value);
            }
        },
        .optional => |info| {
            hash_bytes(hash, "optional");
            hash_type(hash, info.child);
        },
        .pointer => |info| {
            if (info.size != .slice)
                @compileError("PR wire supports slices but not single-item pointers");
            hash_bytes(hash, "slice");
            hash_type(hash, info.child);
        },
        .@"struct" => |info| {
            hash_bytes(hash, "struct");
            inline for (info.fields) |field| {
                hash_bytes(hash, field.name);
                hash_type(hash, field.type);
            }
        },
        .@"union" => |info| {
            hash_bytes(hash, "union");
            const Tag = info.tag_type orelse
                @compileError("PR wire requires tagged unions");
            hash_type(hash, Tag);
            inline for (info.fields) |field| {
                hash_bytes(hash, field.name);
                hash_type(hash, field.type);
            }
        },
        else => @compileError("unsupported PR wire type: " ++ @typeName(T)),
    }
}

fn hash_bytes(hash: *u64, bytes: []const u8) void {
    hash_int(hash, bytes.len);
    for (bytes) |byte| {
        hash.* ^= byte;
        hash.* *%= 1099511628211;
    }
}

fn hash_int(hash: *u64, value: anytype) void {
    var remaining: u64 = @intCast(value);
    for (0..8) |_| {
        hash.* ^= @truncate(remaining);
        hash.* *%= 1099511628211;
        remaining >>= 8;
    }
}

fn make_test_program(backing_allocator: Allocator) !pr.Program {
    var program = pr.Program.init(backing_allocator);
    errdefer program.deinit();
    const arena = program.allocator();

    const scalar_dims = try arena.alloc(i64, 0);
    const scalar = pr.Aval{ .tensor = .{
        .dtype = .f32,
        .shape = .{ .dims = scalar_dims },
    } };

    const literal_op = try arena.create(pr.Op);
    const literal_var = try arena.create(pr.Var);
    literal_var.* = .{
        .id = 0,
        .aval = scalar,
        .defining_op = literal_op,
    };
    const literal_outputs = try arena.alloc(*pr.Var, 1);
    literal_outputs[0] = literal_var;
    literal_op.* = .{
        .id = 4,
        .inputs = try arena.alloc(pr.Operand, 0),
        .outputs = literal_outputs,
        .params = .{ .literal = .{ .f32 = -std.math.inf(f32) } },
    };

    const custom_op = try arena.create(pr.Op);
    const custom_outputs = try arena.alloc(*pr.Var, 2);
    for (custom_outputs, 0..) |*output, index| {
        const variable = try arena.create(pr.Var);
        variable.* = .{
            .id = @intCast(index + 1),
            .aval = scalar,
            .defining_op = custom_op,
        };
        output.* = variable;
    }

    const custom_inputs = try arena.alloc(pr.Operand, 2);
    custom_op.* = .{
        .id = 9,
        .inputs = custom_inputs,
        .outputs = custom_outputs,
        .params = .{ .custom_call = .{
            .target_name = try arena.dupe(u8, "round_trip"),
            .has_side_effect = false,
            .payload = try arena.dupe(u8, "payload"),
        } },
    };
    for (custom_inputs, 0..) |*input, index| {
        input.attach(literal_var, custom_op, @intCast(index));
    }

    const call_op = try arena.create(pr.Op);
    const call_var = try arena.create(pr.Var);
    call_var.* = .{
        .id = 3,
        .aval = scalar,
        .defining_op = call_op,
    };
    const call_inputs = try arena.alloc(pr.Operand, 1);
    const call_outputs = try arena.alloc(*pr.Var, 1);
    call_outputs[0] = call_var;
    call_op.* = .{
        .id = 12,
        .inputs = call_inputs,
        .outputs = call_outputs,
        .params = .{ .call = .{
            .callee = @enumFromInt(1),
        } },
    };
    call_inputs[0].attach(custom_outputs[0], call_op, 0);

    const function_ops = try arena.alloc(*pr.Op, 3);
    function_ops[0] = literal_op;
    function_ops[1] = custom_op;
    function_ops[2] = call_op;
    const returns = try arena.alloc(*pr.Var, 2);
    returns[0] = call_var;
    returns[1] = custom_outputs[1];

    const op_ids = try arena.alloc(u32, 2);
    op_ids[0] = literal_op.id;
    op_ids[1] = custom_op.id;
    const regions = try arena.alloc(pr.Region, 1);
    const region_annotations = try arena.alloc(pr.Annotation, 4);
    region_annotations[0] = @import("transform/outline.zig").annotation;
    region_annotations[1] = @import("kernel").providers_annotation(&.{ "first", "second" });
    region_annotations[2] = .{ .name = "example.priority", .value = .{ .integer = 3 } };
    region_annotations[3] = .{ .name = "example.payload", .value = .{ .bytes = &.{ 0, 127, 255 } } };
    regions[0] = .{
        .id = 7,
        .name = try arena.dupe(u8, "serialized"),
        .annotations = region_annotations,
        .op_ids = op_ids,
    };

    const identity_param = try arena.create(pr.Var);
    identity_param.* = .{
        .id = 0,
        .aval = scalar,
    };
    const identity_params = try arena.alloc(*pr.Var, 1);
    identity_params[0] = identity_param;
    const identity_returns = try arena.alloc(*pr.Var, 1);
    identity_returns[0] = identity_param;

    const functions = try arena.alloc(pr.Function, 2);
    functions[0] = .{
        .name = try arena.dupe(u8, "main"),
        .annotations = &.{.{ .name = "example.function", .value = .{ .boolean = true } }},
        .params = try arena.alloc(*pr.Var, 0),
        .returns = returns,
        .ops = function_ops,
        .regions = regions,
        .var_count = 4,
    };
    functions[1] = .{
        .name = try arena.dupe(u8, "identity"),
        .params = identity_params,
        .returns = identity_returns,
        .ops = try arena.alloc(*pr.Op, 0),
        .regions = try arena.alloc(pr.Region, 0),
        .var_count = 1,
    };
    for (functions, 0..) |func, ordinal| {
        const id = try program.add_function(func);
        std.debug.assert(@intFromEnum(id) == ordinal);
    }
    try program.set_entry(@enumFromInt(0));
    return program;
}

test "binary PR round trip is byte stable" {
    var source = try make_test_program(std.testing.allocator);
    defer source.deinit();
    try pr.validate_program(&source);

    var first: Writer.Allocating = .init(std.testing.allocator);
    defer first.deinit();
    try emit(&source, &first.writer);
    const first_bytes = try first.toOwnedSlice();
    defer std.testing.allocator.free(first_bytes);

    var parsed = try parse(std.testing.allocator, first_bytes);
    defer parsed.deinit();
    try pr.validate_program(&parsed);
    try std.testing.expectEqual(@as(pr.FunctionId, @enumFromInt(0)), try parsed.resolve_entry());

    try std.testing.expectEqual(@as(usize, 2), parsed.functions()[0].ops[1].outputs.len);
    const literal = parsed.functions()[0].ops[0].params.literal.f32;
    try std.testing.expect(std.math.isNegativeInf(literal));
    try std.testing.expect(parsed.functions()[0].find_annotation("example.function").?.value.boolean);

    const literal_op = parsed.functions()[0].ops[0];
    const custom_op = parsed.functions()[0].ops[1];
    const literal_var = literal_op.outputs[0];
    try std.testing.expect(literal_var.defining_op == literal_op);
    try std.testing.expect(literal_var.first_use == &custom_op.inputs[1]);
    try std.testing.expect(custom_op.inputs[0].owner == custom_op);
    try std.testing.expectEqual(@as(u32, 0), custom_op.inputs[0].index);
    try std.testing.expect(custom_op.inputs[0].prev == &custom_op.inputs[1]);
    try std.testing.expect(custom_op.inputs[0].next == null);
    try std.testing.expect(custom_op.inputs[1].prev == null);
    try std.testing.expect(custom_op.inputs[1].next == &custom_op.inputs[0]);
    try std.testing.expectEqualStrings("payload", custom_op.params.custom_call.payload);

    const region = parsed.functions()[0].regions[0];
    try std.testing.expectEqualStrings("serialized", region.name);
    try std.testing.expect(try @import("transform/outline.zig").is_requested(region));
    const providers = (try @import("kernel").requested_providers(region.annotations)).?;
    try std.testing.expectEqual(@as(usize, 2), providers.len());
    try std.testing.expectEqualStrings("first", providers.at(0));
    try std.testing.expectEqualStrings("second", providers.at(1));
    try std.testing.expectEqual(@as(i64, 3), region.find_annotation("example.priority").?.value.integer);
    try std.testing.expectEqualSlices(u8, &.{ 0, 127, 255 }, region.find_annotation("example.payload").?.value.bytes);
    try std.testing.expectEqualSlices(u32, &.{ 4, 9 }, region.op_ids);

    const call_op = parsed.functions()[0].ops[2];
    try std.testing.expectEqual(@as(pr.FunctionId, @enumFromInt(1)), call_op.params.call.callee);
    try std.testing.expectEqual(@as(usize, 1), call_op.outputs.len);
    try std.testing.expect(parsed.functions()[0].returns[0] == call_op.outputs[0]);
    try std.testing.expect(call_op.outputs[0].first_use == null);

    var second: Writer.Allocating = .init(std.testing.allocator);
    defer second.deinit();
    try emit(&parsed, &second.writer);
    const second_bytes = try second.toOwnedSlice();
    defer std.testing.allocator.free(second_bytes);

    try std.testing.expectEqualSlices(u8, first_bytes, second_bytes);
}

test "binary PR preserves inferred entry selection" {
    var source = pr.Program.init(std.testing.allocator);
    defer source.deinit();
    var builder = try pr.FunctionBuilder.init(&source, "only");
    defer builder.deinit();
    _ = try source.add_function(try builder.finish(.{ .returns = &.{} }));

    var encoded: Writer.Allocating = .init(std.testing.allocator);
    defer encoded.deinit();
    try emit(&source, &encoded.writer);
    const bytes = try encoded.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    var parsed = try parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(?pr.FunctionId, null), parsed.entry);
    try std.testing.expectEqual(@as(pr.FunctionId, @enumFromInt(0)), try parsed.resolve_entry());
}

test "binary PR preserves sparse function identities" {
    var source = pr.Program.init(std.testing.allocator);
    defer source.deinit();

    var base_builder = try pr.FunctionBuilder.init(&source, "base");
    defer base_builder.deinit();
    const base_input = try base_builder.param_tensor(.f32, &.{});
    _ = try source.add_function(try base_builder.finish(.{ .returns = &.{base_input} }));

    const checkpoint = source.checkpoint_appends();
    var removed_builder = try pr.FunctionBuilder.init(&source, "removed");
    defer removed_builder.deinit();
    const removed_input = try removed_builder.param_tensor(.f32, &.{});
    _ = try source.add_function(try removed_builder.finish(.{ .returns = &.{removed_input} }));
    source.restore_appends(checkpoint);

    var callee_builder = try pr.FunctionBuilder.init(&source, "callee");
    defer callee_builder.deinit();
    const callee_input = try callee_builder.param_tensor(.f32, &.{});
    const callee_id = try source.add_function(try callee_builder.finish(.{ .returns = &.{callee_input} }));
    try std.testing.expectEqual(@as(pr.FunctionId, @enumFromInt(2)), callee_id);

    var caller_builder = try pr.FunctionBuilder.init(&source, "caller");
    defer caller_builder.deinit();
    const caller_input = try caller_builder.param_tensor(.f32, &.{});
    const call_op = try caller_builder.call(callee_id, &.{caller_input});
    const caller_id = try source.add_function(try caller_builder.finish(.{ .returns = call_op.outputs }));
    try source.set_entry(caller_id);

    var encoded: Writer.Allocating = .init(std.testing.allocator);
    defer encoded.deinit();
    try emit(&source, &encoded.writer);

    var parsed = try parse(std.testing.allocator, encoded.written());
    defer parsed.deinit();
    const function_ids = parsed.function_ids();
    try std.testing.expectEqual(@as(usize, 3), function_ids.len);
    try std.testing.expectEqualSlices(pr.FunctionId, &.{ @enumFromInt(0), @enumFromInt(2), @enumFromInt(3) }, function_ids);
    try std.testing.expectEqual(@as(?pr.FunctionId, @enumFromInt(3)), parsed.entry);
    const parsed_caller = parsed.get_function_by_id(@enumFromInt(3)).?;
    try std.testing.expectEqual(@as(pr.FunctionId, @enumFromInt(2)), parsed_caller.ops[0].params.call.callee);
}

test "convolution parameters round trip" {
    var source = pr.Program.init(std.testing.allocator);
    defer source.deinit();
    var builder = try pr.FunctionBuilder.init(&source, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{ 1, 8, 8, 3 });
    const kernel = try builder.param_tensor(.f32, &.{ 3, 3, 3, 4 });
    const output = try builder.convolution(input, kernel, .{
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
    _ = try source.add_function(try builder.finish(.{ .returns = &.{output} }));

    var encoded: Writer.Allocating = .init(std.testing.allocator);
    defer encoded.deinit();
    try emit(&source, &encoded.writer);
    const bytes = try encoded.toOwnedSlice();
    defer std.testing.allocator.free(bytes);
    var decoded = try parse(std.testing.allocator, bytes);
    defer decoded.deinit();

    const params = decoded.functions()[0].ops[0].params.convolution;
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, params.window_strides);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1, 1 }, params.padding);
    try std.testing.expectEqual(@as(i64, 3), params.dimensions.input_feature_dimension);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, params.dimensions.kernel_spatial_dimensions);
}

test "binary PR rejects version and schema mismatches" {
    var source = try make_test_program(std.testing.allocator);
    defer source.deinit();

    var output: Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try emit(&source, &output.writer);
    const bytes = try output.toOwnedSlice();
    defer std.testing.allocator.free(bytes);

    bytes[magic.len] ^= 1;
    try std.testing.expectError(
        error.UnsupportedVersion,
        parse(std.testing.allocator, bytes),
    );
    bytes[magic.len] ^= 1;

    bytes[magic.len + @sizeOf(u32)] ^= 1;
    try std.testing.expectError(
        error.SchemaMismatch,
        parse(std.testing.allocator, bytes),
    );
}
