//! PR region outlining.
const std = @import("std");
const compilation = @import("compilation");
const pr = @import("../pr.zig");
const pattern = @import("../analysis/pattern.zig");
const region_view = @import("../analysis/region_view.zig");

const Allocator = std.mem.Allocator;

/// Region annotation requesting PR function outlining.
pub const annotation_name = "zigrad.outline";

/// Unit-valued outline request for region builders.
pub const annotation: pr.Annotation = .{
    .name = annotation_name,
    .value = .unit,
};

/// Invalid payloads for the outline annotation contract.
pub const AnnotationError = error{
    /// The annotation value is not a supported outline request.
    InvalidOutlineAnnotation,
};

/// Return whether a region requests outlining.
pub fn is_requested(region: pr.Region) AnnotationError!bool {
    const found = region.find_annotation(annotation_name) orelse return false;
    return switch (found.value) {
        .unit => true,
        .boolean => |value| value,
        else => error.InvalidOutlineAnnotation,
    };
}

/// Configuration for outlining one region.
pub const ApplyOptions = struct {
    /// Unique name assigned to the outlined function.
    ///
    /// Null derives a name from the caller and region id.
    function_name: ?[]const u8 = null,
    /// Annotations attached to the outlined function.
    function_annotations: []const pr.Annotation = &.{},
};

/// Configuration for extracting one operation range.
pub const ExtractOptions = struct {
    /// Unique name assigned to the extracted function.
    ///
    /// Null derives a name from the caller and operation range.
    function_name: ?[]const u8 = null,
    /// Annotations attached to the extracted function.
    function_annotations: []const pr.Annotation = &.{},
    /// Source region omitted from copied metadata. Null omits no region.
    source_region: ?u32 = null,
    /// Validate the complete program after extraction. A caller that disables
    ///  this check must validate after composing its mutations.
    validate: bool = true,
};

/// Configuration for replacing one operation range with a call.
pub const ReplaceOptions = struct {
    /// Source region removed with the operations. Null removes no region.
    source_region: ?u32 = null,
    /// Validate the complete program after replacement. A caller that disables
    ///  this check must validate after composing its mutations.
    validate: bool = true,
};

/// Location of the function and call produced by outlining.
pub const ApplyResult = struct {
    /// Identity of the appended outlined function.
    function_id: pr.FunctionId,
    /// Operation id of the replacement call in the caller.
    call_op_id: u32,
};

/// Structural and allocation failures produced by range outlining.
pub const ApplyError = pr.BuildError || pr.FunctionRegistrationError || error{
    /// The selected function does not exist.
    FunctionIdOutOfRange,
    /// The selected region does not exist.
    RegionNotFound,
    /// The selected region contains no operations.
    EmptyRegion,
    /// A selected region operation does not exist in its function.
    UnknownRegionOperation,
    /// Selected region operations do not form one contiguous range.
    NonContiguousRegion,
    /// A retained nested region partially overlaps the selected range.
    PartialRegionOverlap,
    /// Rebuilding referenced a source value before binding its replacement.
    MissingValueMapping,
    /// The selected range is empty or outside its function.
    InvalidOperationRange,
};

const OpRange = struct {
    start: usize,
    end: usize,
};

/// Extract an operation range into a callable function without changing its caller.
pub fn extract_range(
    /// Program that owns the source and extracted functions.
    program: *pr.Program,
    /// Allocator used for temporary maps and boundary analysis.
    scratch: Allocator,
    /// Function containing `range`.
    function_id: pr.FunctionId,
    /// Nonempty operation range to extract.
    range: pattern.Range,
    /// Naming, annotations, and nested-region handling.
    opts: ExtractOptions,
) ApplyError!pr.FunctionId {
    const source = program.get_function_by_id(function_id) orelse
        return error.FunctionIdOutOfRange;
    if (range.start >= range.end or range.end > source.ops.len)
        return error.InvalidOperationRange;

    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    const generated_base = if (opts.function_name == null)
        try scratch.print("{s}_candidate_{d}_{d}", .{ source.name, range.start, range.end })
    else
        null;
    defer if (generated_base) |name| scratch.free(name);
    const function_name = opts.function_name orelse
        try program.reserve_unique_function_name(generated_base.?);
    if (program.get_function_id(function_name) != null)
        return error.DuplicateFunctionName;

    const op_ids = try scratch.alloc(u32, range.len());
    defer scratch.free(op_ids);
    for (source.ops[range.start..range.end], op_ids) |op, *op_id|
        op_id.* = op.id;
    const target = pr.Region{
        .id = std.math.maxInt(u32),
        .name = function_name,
        .annotations = &.{},
        .op_ids = op_ids,
    };
    const desc = try region_view.describe(scratch, source, target);
    defer desc.deinit(scratch);

    const callee_op_ids = try scratch.alloc(?u32, source.ops.len);
    defer scratch.free(callee_op_ids);
    @memset(callee_op_ids, null);

    const op_range = OpRange{ .start = range.start, .end = range.end };
    var extracted = try build_outlined_function(
        program,
        scratch,
        source,
        desc,
        op_range,
        function_name,
        opts.function_annotations,
        callee_op_ids,
    );
    extracted.regions = try build_outlined_regions(
        program.allocator(),
        source,
        opts.source_region,
        op_range,
        callee_op_ids,
    );

    const extracted_id = try program.add_function(extracted);
    if (opts.validate) try pr.validate_program(program);
    return extracted_id;
}

/// Dense source-to-replacement variable map used while rebuilding a function.
const VarRemap = struct {
    /// Owns `replacements` and slices returned by allocation methods.
    allocator: Allocator,
    /// Replacement values indexed by source variable id. Null marks an
    ///  unbound source variable.
    replacements: []?*pr.Var,

    /// Allocate an empty map for `var_count` source variables.
    fn init(allocator: Allocator, var_count: u32) Allocator.Error!VarRemap {
        const replacements = try allocator.alloc(?*pr.Var, var_count);
        @memset(replacements, null);
        return .{ .allocator = allocator, .replacements = replacements };
    }

    /// Release replacement storage.
    fn deinit(self: VarRemap) void {
        self.allocator.free(self.replacements);
    }

    /// Bind an unbound source variable to its replacement.
    fn bind(self: VarRemap, source: *const pr.Var, replacement: *pr.Var) void {
        std.debug.assert(source.id < self.replacements.len);
        std.debug.assert(self.replacements[source.id] == null);
        self.replacements[source.id] = replacement;
    }

    /// Bind corresponding source and replacement variables.
    fn bind_all(self: VarRemap, sources: []const *pr.Var, replacements: []const *pr.Var) void {
        std.debug.assert(sources.len == replacements.len);
        for (sources, replacements) |source, replacement| self.bind(source, replacement);
    }

    /// Resolve a bound source variable.
    fn resolve(self: VarRemap, source: *const pr.Var) ApplyError!*pr.Var {
        std.debug.assert(source.id < self.replacements.len);
        return self.replacements[source.id] orelse error.MissingValueMapping;
    }

    /// Allocate resolved replacements for `sources`.
    fn alloc_values(self: VarRemap, sources: []const *pr.Var) ApplyError![]*pr.Var {
        const resolved = try self.allocator.alloc(*pr.Var, sources.len);
        errdefer self.allocator.free(resolved);
        for (sources, resolved) |source, *replacement|
            replacement.* = try self.resolve(source);
        return resolved;
    }

    /// Allocate resolved replacements for the inputs of `op`.
    fn alloc_inputs(self: VarRemap, op: *const pr.Op) ApplyError![]*pr.Var {
        const resolved = try self.allocator.alloc(*pr.Var, op.inputs.len);
        errdefer self.allocator.free(resolved);
        for (op.inputs, resolved) |operand, *replacement|
            replacement.* = try self.resolve(operand.value);
        return resolved;
    }
};

/// Outline one region into a PR function and replace its operations with a call.
///
/// The program arena owns the resulting function, call, annotations, and remapped regions.
pub fn apply(
    /// Program containing the selected region.
    program: *pr.Program,
    /// Allocator used for temporary maps and boundary analysis.
    scratch: Allocator,
    /// Function containing the selected region.
    function_id: pr.FunctionId,
    /// Region to outline and replace.
    region_id: u32,
    /// Naming and annotation configuration.
    opts: ApplyOptions,
) ApplyError!ApplyResult {
    const source = program.get_function_by_id(function_id) orelse return error.FunctionIdOutOfRange;
    const target = find_region(source, region_id) orelse return error.RegionNotFound;
    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    const generated_base = if (opts.function_name == null)
        try scratch.print("{s}_outlined_{d}", .{ source.name, target.id })
    else
        null;
    defer if (generated_base) |name| scratch.free(name);
    const function_name = opts.function_name orelse
        try program.reserve_unique_function_name(generated_base.?);
    if (program.get_function_id(function_name) != null) return error.DuplicateFunctionName;
    const target_range = try region_range(source, target);
    try validate_region_overlaps(source, target_range, target.id);

    const desc = try region_view.describe(scratch, source, target);
    defer desc.deinit(scratch);

    const callee_op_ids = try scratch.alloc(?u32, source.ops.len);
    defer scratch.free(callee_op_ids);
    @memset(callee_op_ids, null);

    var outlined = try build_outlined_function(
        program,
        scratch,
        source,
        desc,
        target_range,
        function_name,
        opts.function_annotations,
        callee_op_ids,
    );
    outlined.regions = try build_outlined_regions(
        program.allocator(),
        source,
        target.id,
        target_range,
        callee_op_ids,
    );

    const outlined_id = try program.add_function(outlined);
    const call_op_id = try replace_range(
        program,
        scratch,
        function_id,
        .{ .start = target_range.start, .end = target_range.end },
        outlined_id,
        .{ .source_region = target.id },
    );
    return .{
        .function_id = outlined_id,
        .call_op_id = call_op_id,
    };
}

/// Replace an operation range with a call to an existing function.
pub fn replace_range(
    /// Program containing the caller and callee.
    program: *pr.Program,
    /// Allocator used for temporary maps and boundary analysis.
    scratch: Allocator,
    /// Function containing `range`.
    function_id: pr.FunctionId,
    /// Nonempty operation range to replace.
    range: pattern.Range,
    /// Registered function whose signature matches the range boundary.
    callee_id: pr.FunctionId,
    /// Nested-region handling for the removed operations.
    opts: ReplaceOptions,
) ApplyError!u32 {
    const source = program.get_function_by_id(function_id) orelse
        return error.FunctionIdOutOfRange;
    if (range.start >= range.end or range.end > source.ops.len)
        return error.InvalidOperationRange;
    const target_range = OpRange{ .start = range.start, .end = range.end };
    try validate_region_overlaps(source, target_range, opts.source_region);

    const op_ids = try scratch.alloc(u32, range.len());
    defer scratch.free(op_ids);
    for (source.ops[range.start..range.end], op_ids) |op, *op_id|
        op_id.* = op.id;
    const target = pr.Region{
        .id = opts.source_region orelse std.math.maxInt(u32),
        .name = source.name,
        .annotations = &.{},
        .op_ids = op_ids,
    };
    const desc = try region_view.describe(scratch, source, target);
    defer desc.deinit(scratch);

    const caller_op_ids = try scratch.alloc(?u32, source.ops.len);
    defer scratch.free(caller_op_ids);
    @memset(caller_op_ids, null);

    var caller = try build_caller_function(
        program,
        scratch,
        source,
        desc,
        target_range,
        callee_id,
        caller_op_ids,
    );
    const call_op_id = caller_op_ids[target_range.start].?;
    caller.annotations = source.annotations;
    caller.regions = try build_caller_regions(
        program.allocator(),
        source,
        opts.source_region,
        target_range,
        call_op_id,
        caller_op_ids,
    );

    program.replace_function(function_id, caller) catch unreachable;
    errdefer program.replace_function(function_id, source) catch unreachable;
    if (opts.validate) try pr.validate_program(program);
    return call_op_id;
}

/// Compilation operation that consumes explicit outline annotations.
pub const Pass = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    pub fn run(_: Pass, program: Input, ctx: *compilation.Context) !Output {
        while (try find_innermost_request(program)) |request| {
            const remaining = try annotations_without(
                ctx.allocator,
                request.region.annotations,
                annotation_name,
            );
            defer ctx.allocator.free(remaining);

            _ = try apply(
                program,
                ctx.allocator,
                request.function_id,
                request.region.id,
                .{
                    .function_annotations = remaining,
                },
            );
        }
        return program;
    }
};

const Request = struct {
    function_id: pr.FunctionId,
    region: pr.Region,
};

fn find_innermost_request(program: *const pr.Program) AnnotationError!?Request {
    var found: ?Request = null;
    for (program.functions(), program.function_ids()) |func, function_id| {
        for (func.regions) |region| {
            if (!try is_requested(region)) continue;
            if (found == null or region.op_ids.len < found.?.region.op_ids.len) {
                found = .{ .function_id = function_id, .region = region };
            }
        }
    }
    return found;
}

fn annotations_without(
    allocator: Allocator,
    annotations: []const pr.Annotation,
    name: []const u8,
) Allocator.Error![]const pr.Annotation {
    var result = try std.ArrayList(pr.Annotation).initCapacity(allocator, annotations.len);
    defer result.deinit(allocator);
    for (annotations) |item| {
        if (!std.mem.eql(u8, item.name, name)) try result.append(allocator, item);
    }
    return try result.toOwnedSlice(allocator);
}

fn find_region(func: pr.Function, region_id: u32) ?pr.Region {
    for (func.regions) |region| {
        if (region.id == region_id) return region;
    }
    return null;
}

fn region_range(func: pr.Function, region: pr.Region) ApplyError!OpRange {
    if (region.op_ids.len == 0) return error.EmptyRegion;
    const start = func.op_index_by_id(region.op_ids[0]) orelse return error.UnknownRegionOperation;
    const end = start + region.op_ids.len;
    if (end > func.ops.len) return error.NonContiguousRegion;
    for (region.op_ids, 0..) |op_id, offset| {
        if (func.ops[start + offset].id != op_id) return error.NonContiguousRegion;
    }
    return .{ .start = start, .end = end };
}

fn validate_region_overlaps(
    source: pr.Function,
    target_range: OpRange,
    source_region: ?u32,
) ApplyError!void {
    for (source.regions) |region| {
        if (region.id == source_region) continue;
        const candidate_range = try region_range(source, region);
        if (ranges_disjoint(target_range, candidate_range)) continue;
        if (range_contains(target_range, candidate_range)) continue;
        if (range_contains(candidate_range, target_range)) continue;
        return error.PartialRegionOverlap;
    }
}

fn ranges_disjoint(lhs: OpRange, rhs: OpRange) bool {
    return lhs.end <= rhs.start or rhs.end <= lhs.start;
}

fn range_contains(outer: OpRange, inner: OpRange) bool {
    return outer.start <= inner.start and outer.end >= inner.end;
}

fn build_outlined_function(
    program: *pr.Program,
    scratch: Allocator,
    source: pr.Function,
    desc: region_view.RegionView,
    target_range: OpRange,
    function_name: []const u8,
    function_annotations: []const pr.Annotation,
    op_ids: []?u32,
) ApplyError!pr.Function {
    var builder = try pr.FunctionBuilder.init(program, function_name);
    defer builder.deinit();

    const vars = try VarRemap.init(scratch, source.var_count);
    defer vars.deinit();

    for (desc.inputs) |input| {
        vars.bind(input, try builder.param_like(input.aval));
    }

    for (source.ops[target_range.start..target_range.end], target_range.start..) |op, source_index| {
        const inputs = try vars.alloc_inputs(op);
        defer scratch.free(inputs);
        const replayed = try builder.replay_op(op, inputs);
        op_ids[source_index] = replayed.id;
        vars.bind_all(op.outputs, replayed.outputs);
    }

    const returns = try vars.alloc_values(desc.outputs);
    defer scratch.free(returns);
    var outlined = try builder.finish(.{ .returns = returns });
    outlined.annotations = try pr.dupe_annotations(program.allocator(), function_annotations);
    return outlined;
}

fn build_caller_function(
    program: *pr.Program,
    scratch: Allocator,
    source: pr.Function,
    desc: region_view.RegionView,
    target_range: OpRange,
    callee_id: pr.FunctionId,
    op_ids: []?u32,
) ApplyError!pr.Function {
    var builder = try pr.FunctionBuilder.init(program, source.name);
    defer builder.deinit();

    const vars = try VarRemap.init(scratch, source.var_count);
    defer vars.deinit();

    for (source.params) |param| {
        vars.bind(param, try builder.param_like(param.aval));
    }

    var source_index: usize = 0;
    while (source_index < source.ops.len) {
        if (source_index == target_range.start) {
            const inputs = try vars.alloc_values(desc.inputs);
            defer scratch.free(inputs);
            const call_op = try builder.call(callee_id, inputs);
            vars.bind_all(desc.outputs, call_op.outputs);
            op_ids[source_index] = call_op.id;
            source_index = target_range.end;
            continue;
        }

        const op = source.ops[source_index];
        const inputs = try vars.alloc_inputs(op);
        defer scratch.free(inputs);
        const replayed = try builder.replay_op(op, inputs);
        op_ids[source_index] = replayed.id;
        vars.bind_all(op.outputs, replayed.outputs);
        source_index += 1;
    }

    const returns = try vars.alloc_values(source.returns);
    defer scratch.free(returns);
    return try builder.finish(.{ .returns = returns });
}

fn build_outlined_regions(
    arena: Allocator,
    source: pr.Function,
    excluded_region_id: ?u32,
    target_range: OpRange,
    op_ids: []const ?u32,
) ApplyError![]const pr.Region {
    var regions = std.ArrayList(pr.Region).empty;
    for (source.regions) |region| {
        if (region.id == excluded_region_id) continue;
        const candidate_range = try region_range(source, region);
        if (!range_contains(target_range, candidate_range)) continue;
        try regions.append(arena, try remap_region(arena, source, region, op_ids, null));
    }
    return try regions.toOwnedSlice(arena);
}

fn build_caller_regions(
    arena: Allocator,
    source: pr.Function,
    source_region: ?u32,
    target_range: OpRange,
    call_op_id: u32,
    op_ids: []const ?u32,
) ApplyError![]const pr.Region {
    var regions = std.ArrayList(pr.Region).empty;
    for (source.regions) |region| {
        if (region.id == source_region) continue;
        const candidate_range = try region_range(source, region);
        if (range_contains(target_range, candidate_range)) continue;
        const replacement = if (range_contains(candidate_range, target_range)) call_op_id else null;
        try regions.append(arena, try remap_region(arena, source, region, op_ids, replacement));
    }
    return try regions.toOwnedSlice(arena);
}

fn remap_region(
    arena: Allocator,
    source: pr.Function,
    region: pr.Region,
    op_ids: []const ?u32,
    replacement: ?u32,
) ApplyError!pr.Region {
    var mapped = std.ArrayList(u32).empty;
    var wrote_replacement = false;
    for (region.op_ids) |op_id| {
        const index = source.op_index_by_id(op_id) orelse return error.UnknownRegionOperation;
        if (op_ids[index]) |new_id| {
            try mapped.append(arena, new_id);
            if (replacement != null and new_id == replacement.?) wrote_replacement = true;
        } else if (!wrote_replacement) {
            try mapped.append(arena, replacement orelse return error.MissingValueMapping);
            wrote_replacement = true;
        }
    }
    return .{
        .id = region.id,
        .name = region.name,
        .annotations = region.annotations,
        .op_ids = try mapped.toOwnedSlice(arena),
    };
}

test is_requested {
    const requested = pr.Region{
        .id = 0,
        .name = "requested",
        .annotations = &.{annotation},
        .op_ids = &.{},
    };
    try @import("std").testing.expect(try is_requested(requested));
}

test "apply outlines a nested region and preserves surrounding regions" {
    const testing = std.testing;
    const retained = pr.Annotation{
        .name = "example.payload.v1",
        .value = .{ .bytes = &.{ 1, 2, 3 } },
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const x = try builder.param_tensor(.f32, &.{ 2, 2 });
    const y = try builder.param_tensor(.f32, &.{ 2, 2 });
    try builder.push_region("outer", &.{.{ .name = "example.outer", .value = .unit }});
    const sum = try builder.add(x, y);
    try builder.push_region("target", &.{ annotation, retained });
    try builder.push_region("inner", &.{.{ .name = "example.inner", .value = .unit }});
    const product = try builder.multiply(sum, y);
    try builder.pop_region();
    const logged = try builder.log(product);
    try builder.pop_region();
    const result = try builder.add(logged, x);
    try builder.pop_region();

    const main = try builder.finish(.{ .returns = &.{result} });
    const main_id = try program.add_function(main);

    const target = for (main.regions) |region| {
        if (std.mem.eql(u8, region.name, "target")) break region;
    } else unreachable;
    const outlined = try apply(
        &program,
        testing.allocator,
        main_id,
        target.id,
        .{
            .function_name = "main_target",
            .function_annotations = &.{retained},
        },
    );

    try testing.expectEqual(@as(pr.FunctionId, @fromBackingInt(@intCast(1))), outlined.function_id);
    try testing.expectEqual(@as(u32, 1), outlined.call_op_id);
    try testing.expectEqual(@as(usize, 2), program.functions().len);

    const caller = program.functions()[0];
    try testing.expectEqual(@as(usize, 3), caller.ops.len);
    try testing.expectEqual(pr.Prim.call, caller.ops[1].prim());
    try testing.expectEqual(@as(pr.FunctionId, @fromBackingInt(@intCast(1))), caller.ops[1].params.call.callee);
    try testing.expectEqual(@as(usize, 2), caller.ops[1].inputs.len);
    try testing.expectEqual(@as(usize, 1), caller.ops[1].outputs.len);
    try testing.expectEqual(@as(usize, 1), caller.regions.len);
    try testing.expectEqualStrings("outer", caller.regions[0].name);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, caller.regions[0].op_ids);

    const callee = program.functions()[1];
    try testing.expectEqualStrings("main_target", callee.name);
    try testing.expectEqualSlices(
        u8,
        &.{ 1, 2, 3 },
        callee.find_annotation("example.payload.v1").?.value.bytes,
    );
    try testing.expectEqual(@as(usize, 2), callee.params.len);
    try testing.expectEqual(@as(usize, 2), callee.ops.len);
    try testing.expectEqual(@as(usize, 1), callee.returns.len);
    try testing.expectEqual(@as(usize, 1), callee.regions.len);
    try testing.expectEqualStrings("inner", callee.regions[0].name);
    try testing.expectEqualSlices(u32, &.{0}, callee.regions[0].op_ids);
}

test "Pass consumes outline and transfers independent annotations" {
    const kernel = @import("kernel");
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    try builder.push_region("matmul", &.{
        annotation,
        .{ .name = "example.payload.v1", .value = .{ .bytes = &.{ 4, 5, 6 } } },
        kernel.provider_annotation("tvm"),
    });
    const result = try builder.mm(lhs, rhs);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{result} }));

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (Pass{}).run(&program, &ctx);

    try testing.expectEqual(@as(usize, 2), program.functions().len);
    try testing.expectEqual(@as(usize, 0), program.functions()[0].regions.len);
    const callee = program.functions()[1];
    try testing.expect(callee.find_annotation(annotation_name) == null);
    try testing.expectEqualSlices(
        u8,
        &.{ 4, 5, 6 },
        callee.find_annotation("example.payload.v1").?.value.bytes,
    );
    try testing.expectEqualStrings(
        "tvm",
        callee.find_annotation(kernel.provider_annotation_name).?.value.as_string().?,
    );
}

test "apply preserves multiple region outputs" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const x = try builder.param_tensor(.f32, &.{2});
    const y = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("pair", &.{annotation});
    const exponential = try builder.exp(x);
    const logarithm = try builder.log(y);
    try builder.pop_region();

    const main = try builder.finish(.{ .returns = &.{ exponential, logarithm } });
    const main_id = try program.add_function(main);
    _ = try apply(&program, testing.allocator, main_id, main.regions[0].id, .{
        .function_name = "main_pair",
    });

    try testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);
    try testing.expectEqual(@as(usize, 2), program.functions()[0].ops[0].outputs.len);
    try testing.expectEqual(@as(usize, 2), program.functions()[1].returns.len);
}
