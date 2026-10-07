//! PR kernel-provider substitution.
//!
//! `ExtractCandidates` turns provider ranges into callable PR functions.
//! `KernelizePass` queries a populated `KernelStore` and substitutes ranges
//!  whose selections supply provider artifacts.
//!
//! An unreplaced candidate leaves the source graph unchanged.
//!
//! Provider compilation and tuning run after extraction and before substitution.
const std = @import("std");
const compilation = @import("compilation");
const device = @import("device");
const output_mod = @import("output");
const effects = @import("../analysis/effects.zig");
const fingerprint = @import("../analysis/fingerprint.zig");
const pattern = @import("../analysis/pattern.zig");
const outline = @import("outline.zig");
const pr = @import("../pr.zig");
const kernel = @import("kernel");

const log = std.log.scoped(.@"zg/kernelize");

const KernelEntryOutcome = enum { provider, unreplaced };

/// Diagnostic record for one function encountered during kernelization.
///
/// `name` and `provider` borrow storage that must outlive the report.
///
/// The report arena stores `ops` and `shape` until `Report.deinit`.
const KernelEntry = struct {
    name: []const u8,
    provider: []const u8,
    ops: []const u8,
    shape: []const u8,
    outcome: KernelEntryOutcome,
};

/// Diagnostics collected while kernelizing a program.
pub const Report = struct {
    /// Owns generated diagnostic strings and entry storage.
    arena: std.heap.ArenaAllocator,
    /// Diagnostics in candidate processing order.
    entries: std.ArrayList(KernelEntry) = .empty,

    /// Initialize an empty kernelization report.
    pub fn init(allocator: std.mem.Allocator) Report {
        return .{ .arena = .init(allocator) };
    }

    /// Release diagnostic strings and entries.
    pub fn deinit(self: *Report) void {
        self.entries.deinit(self.arena.allocator());
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Emits diagnostics collected by a preceding kernelization operation.
pub const DumpKernels = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Report produced by kernelization.
    report: *const Report,
    /// Output destination and format.
    target: output_mod.Target,

    pub fn run(self: DumpKernels, program: Input, ctx: *compilation.Context) !Output {
        if (self.report.entries.items.len == 0) return program;

        const task = struct {
            entries: []const KernelEntry,

            pub fn emit(value: @This(), writer: *std.Io.Writer) !void {
                try dump_kernel_entries(writer, value.entries);
            }
        }{ .entries = self.report.entries.items };
        try output_mod.write(ctx.io, self.target, task);
        return program;
    }
};

/// Extracts discovered candidates as callables without changing their source.
pub const ExtractCandidates = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Discovered source boundaries to extract.
    candidates: *const kernel.Candidates,
    /// Destination replaced transactionally with the extracted collection.
    extracted: *kernel.ExtractedCandidates,

    pub fn run(self: ExtractCandidates, program: Input, ctx: *compilation.Context) !Output {
        std.debug.assert(self.extracted.entries.items.len == 0);
        var result = kernel.ExtractedCandidates.init(self.extracted.allocator);
        errdefer result.deinit();
        const saved = program.checkpoint_appends();
        errdefer program.restore_appends(saved);
        for (self.candidates.entries.items) |candidate| {
            const annotations = try candidate_annotations(program, candidate);
            const function_annotations = try without_provider_annotation(
                ctx.allocator,
                annotations,
            );
            defer ctx.allocator.free(function_annotations);
            const source = program.get_function_by_id(candidate.source_function) orelse
                return error.CallUnresolvedCallee;
            const range = candidate.resolve_range(source) orelse
                return error.StaleKernelCandidate;
            const function_id = try outline.extract_range(
                program,
                ctx.allocator,
                candidate.source_function,
                range,
                .{
                    .function_annotations = function_annotations,
                    .source_region = candidate.explicit_region,
                    .validate = false,
                },
            );
            try remove_provider_annotations(program, function_id);
            try result.append(candidate, function_id);
        }
        if (self.candidates.entries.items.len > 0) try pr.validate_program(program);
        self.extracted.deinit();
        self.extracted.* = result;
        return program;
    }
};

/// Discovers provider-supported PR ranges without changing the program.
///
/// Explicit requests and opportunistic matches are additive. Equal boundaries
///  merge their provider lists, while other overlaps remain separate entries.
/// Every explicitly requested provider must be configured for this operation.
pub const DiscoverCandidates = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Providers available to explicit requests and opportunistic discovery.
    providers: []const kernel.KernelProvider,
    /// Destination populated with discovered boundaries.
    candidates: *kernel.Candidates,

    pub fn run(self: DiscoverCandidates, program: Input, ctx: *compilation.Context) !Output {
        std.debug.assert(self.candidates.entries.items.len == 0);
        try kernel.validate_providers(self.providers);
        var result = kernel.Candidates.init(self.candidates.allocator);
        errdefer result.deinit();
        for (program.functions(), program.function_ids()) |func, function_id| {
            try discover_explicit(func, function_id, self.providers, &result);
            try discover_function(
                ctx.allocator,
                func,
                function_id,
                self.providers,
                &result,
            );
        }
        self.candidates.deinit();
        self.candidates.* = result;
        return program;
    }
};

fn discover_function(
    allocator: std.mem.Allocator,
    func: pr.Function,
    function_id: pr.FunctionId,
    providers: []const kernel.KernelProvider,
    candidates: *kernel.Candidates,
) !void {
    var matches = std.ArrayList(kernel.Match).empty;
    defer matches.deinit(allocator);
    for (providers) |*provider| {
        matches.clearRetainingCapacity();
        try provider.compiler.discover(func, &matches, allocator);
        for (matches.items) |matched| {
            if (matched.start >= matched.end or matched.end > func.ops.len)
                return error.InvalidKernelMatch;
            try candidates.add(
                function_id,
                func,
                matched,
                provider.name,
                null,
            );
        }
    }
}

fn discover_explicit(
    func: pr.Function,
    function_id: pr.FunctionId,
    providers: []const kernel.KernelProvider,
    candidates: *kernel.Candidates,
) !void {
    for (func.regions) |region| {
        const request = (try kernel.requested_providers(region.annotations)) orelse continue;
        const range = try region_range(func, region);
        for (0..request.len()) |provider_index| {
            const provider_name = request.at(provider_index);
            if (kernel.find_provider(providers, provider_name) != null) continue;
            log.info("region '{s}' in function '{s}' requests unconfigured provider '{s}'", .{
                region.name,
                func.name,
                provider_name,
            });
            return error.ProviderNotConfigured;
        }
        for (0..request.len()) |provider_index| {
            try candidates.add(
                function_id,
                func,
                range,
                request.at(provider_index),
                region.id,
            );
        }
    }
}

fn region_range(func: pr.Function, region: pr.Region) !pattern.Range {
    if (region.op_ids.len == 0) return error.InvalidKernelMatch;
    var start = func.ops.len;
    var end: usize = 0;
    for (region.op_ids) |op_id| {
        const index = func.op_index_by_id(op_id) orelse return error.InvalidKernelMatch;
        start = @min(start, index);
        end = @max(end, index + 1);
    }
    if (end - start != region.op_ids.len) return error.InvalidKernelMatch;
    return .{ .start = start, .end = end };
}

fn ranges_overlap(existing: pattern.Range, matched: pattern.Range) bool {
    return existing.start < matched.end and matched.start < existing.end;
}

fn candidate_annotations(
    program: *const pr.Program,
    candidate: kernel.CandidateRegion,
) ![]const pr.Annotation {
    const region_id = candidate.explicit_region orelse return &.{};
    const func = program.get_function_by_id(candidate.source_function) orelse
        return error.CallUnresolvedCallee;
    for (func.regions) |region| {
        if (region.id == region_id) return region.annotations;
    }
    return error.InvalidKernelMatch;
}

fn without_provider_annotation(
    allocator: std.mem.Allocator,
    annotations: []const pr.Annotation,
) std.mem.Allocator.Error![]const pr.Annotation {
    var result = try std.ArrayList(pr.Annotation).initCapacity(allocator, annotations.len);
    for (annotations) |annotation| {
        if (!std.mem.eql(u8, annotation.name, kernel.provider_annotation_name))
            try result.append(allocator, annotation);
    }
    return try result.toOwnedSlice(allocator);
}

fn remove_provider_annotations(
    program: *pr.Program,
    function_id: pr.FunctionId,
) std.mem.Allocator.Error!void {
    var func = program.get_function_by_id(function_id) orelse unreachable;
    func.annotations = try without_provider_annotation(
        program.allocator(),
        func.annotations,
    );
    const regions = try program.allocator().alloc(pr.Region, func.regions.len);
    for (func.regions, regions) |source, *region| {
        region.* = source;
        region.annotations = try without_provider_annotation(
            program.allocator(),
            source.annotations,
        );
    }
    func.regions = regions;
    program.replace_function(function_id, func) catch unreachable;
}

/// Kernelization pass state.
///
/// Consults a pre-computed `KernelStore`, materializes selected non-overlapping
///  candidates, and replaces their calls with `custom_call` ops.
pub const KernelizePass = struct {
    pub const Input = *pr.Program;
    pub const Output = *pr.Program;

    /// Pre-computed selections populated before this operation runs, borrowed.
    store: *const kernel.KernelStore,

    /// Extracted candidates corresponding to store selections, borrowed.
    candidates: *const kernel.ExtractedCandidates,

    /// Device used when the selection store was populated.
    device: device.Device,

    /// Optional destination for diagnostics consumed by `DumpKernels`.
    report: ?*Report = null,

    pub fn run(
        self: *KernelizePass,
        program: Input,
        ctx: *compilation.Context,
    ) !Output {
        try self.run_program(program, ctx.allocator);
        return program;
    }

    fn run_program(
        self: *KernelizePass,
        program: *pr.Program,
        allocator: std.mem.Allocator,
    ) !void {
        var decisions = std.ArrayList(Decision).empty;
        defer {
            for (decisions.items) |decision| allocator.free(decision.selection_key);
            decisions.deinit(allocator);
        }
        for (self.candidates.entries.items) |candidate| {
            const boundary = candidate.boundary;
            const callable = program.get_function_by_id(candidate.callable_function) orelse
                return error.CallUnresolvedCallee;
            const source = program.get_function_by_id(boundary.source_function) orelse
                return error.CallUnresolvedCallee;
            const range = boundary.resolve_range(source) orelse
                return error.StaleKernelCandidate;
            const function_fingerprint = try fingerprint.function(allocator, callable);
            const key = try kernel.make_selection_key(
                allocator,
                boundary,
                self.device,
                function_fingerprint,
            );
            errdefer allocator.free(key.bytes);
            try decisions.append(allocator, .{
                .candidate = candidate,
                .range = range,
                .selection_key = key.bytes,
                .selection = self.store.get(key),
            });
        }

        try validate_selected_nonoverlap(decisions.items);
        var snapshots = try std.ArrayList(FunctionSnapshot).initCapacity(
            allocator,
            decisions.items.len,
        );
        defer snapshots.deinit(allocator);
        for (decisions.items) |decision| {
            if (!decision_uses_provider(decision)) continue;
            for (snapshots.items) |snapshot| {
                if (snapshot.id == decision.candidate.boundary.source_function) break;
            } else {
                const source = program.get_function_by_id(
                    decision.candidate.boundary.source_function,
                ) orelse
                    return error.CallUnresolvedCallee;
                snapshots.appendAssumeCapacity(.{
                    .id = decision.candidate.boundary.source_function,
                    .function = source,
                });
            }
        }
        const report_len = if (self.report) |report| report.entries.items.len else 0;
        errdefer {
            for (snapshots.items) |snapshot|
                program.replace_function(snapshot.id, snapshot.function) catch unreachable;
            if (self.report) |report| report.entries.shrinkRetainingCapacity(report_len);
        }
        std.mem.sortUnstable(
            Decision,
            decisions.items,
            {},
            decision_descending,
        );

        for (decisions.items) |decision| {
            const candidate = decision.candidate;
            const callable = program.get_function_by_id(candidate.callable_function) orelse
                return error.CallUnresolvedCallee;
            const selected_provider = if (decision.selection) |selection| switch (selection.candidate) {
                .provider => |stored| stored.provider_name,
                .unreplaced => null,
            } else null;

            if (selected_provider) |provider_name| {
                try materialize_candidate(
                    program,
                    allocator,
                    candidate,
                    decision.range,
                    callable,
                    decision.selection_key,
                );
                log.debug("selected provider '{s}' for function '{s}'", .{
                    provider_name,
                    callable.name,
                });
            } else if (decision.selection) |selection| {
                log.debug("left function '{s}' unreplaced: {s}", .{
                    callable.name,
                    selection.reason,
                });
            }

            if (self.report) |report| {
                const report_allocator = report.arena.allocator();
                try report.entries.append(report_allocator, .{
                    .name = callable.name,
                    .provider = selected_provider orelse try build_providers_str(
                        report_allocator,
                        candidate.boundary.request(),
                    ),
                    .ops = try build_ops_str(report_allocator, callable),
                    .shape = try build_shape_str(report_allocator, callable),
                    .outcome = if (selected_provider == null) .unreplaced else .provider,
                });
            }
        }
        if (snapshots.items.len > 0) try pr.validate_program(program);
    }

    fn materialize_candidate(
        program: *pr.Program,
        scratch: std.mem.Allocator,
        candidate: kernel.ExtractedCandidate,
        range: pattern.Range,
        callable: pr.Function,
        selection_key: []const u8,
    ) !void {
        const call_op_id = try outline.replace_range(
            program,
            scratch,
            candidate.boundary.source_function,
            range,
            candidate.callable_function,
            .{
                .source_region = candidate.boundary.explicit_region,
                .validate = false,
            },
        );

        const source = program.get_function_by_id(candidate.boundary.source_function) orelse
            return error.CallUnresolvedCallee;
        const call = source.op_by_id(call_op_id) orelse
            return error.CallUnresolvedCallee;
        try rewrite_call(
            program.allocator(),
            call,
            selection_key,
            effects.function_may_have_side_effects(program, callable),
        );
    }

    fn rewrite_call(
        arena: std.mem.Allocator,
        op: *pr.Op,
        selection_key: []const u8,
        has_side_effect: bool,
    ) std.mem.Allocator.Error!void {
        op.params = .{ .custom_call = .{
            .target_name = try arena.dupe(u8, kernel.dispatch_target_name),
            .has_side_effect = has_side_effect,
            .payload = try arena.dupe(u8, selection_key),
        } };
    }
};

const Decision = struct {
    candidate: kernel.ExtractedCandidate,
    range: pattern.Range,
    selection_key: []const u8,
    selection: ?kernel.Selection,
};

const FunctionSnapshot = struct {
    id: pr.FunctionId,
    function: pr.Function,
};

fn decision_descending(
    _: void,
    lhs: Decision,
    rhs: Decision,
) bool {
    const left = lhs.candidate;
    const right = rhs.candidate;
    if (left.boundary.source_function != right.boundary.source_function)
        return @backingInt(left.boundary.source_function) >
            @backingInt(right.boundary.source_function);
    return lhs.range.start > rhs.range.start;
}

fn validate_selected_nonoverlap(decisions: []const Decision) !void {
    for (decisions, 0..) |decision, index| {
        if (!decision_uses_provider(decision)) continue;
        const candidate = decision.candidate.boundary;
        for (decisions[0..index]) |prior_decision| {
            if (!decision_uses_provider(prior_decision)) continue;
            const prior = prior_decision.candidate.boundary;
            if (candidate.source_function == prior.source_function and
                ranges_overlap(decision.range, prior_decision.range))
            {
                return error.OverlappingKernelSelections;
            }
        }
    }
}

fn decision_uses_provider(decision: Decision) bool {
    const selection = decision.selection orelse return false;
    return switch (selection.candidate) {
        .provider => true,
        .unreplaced => false,
    };
}

// Kernel Dump Helpers

fn build_ops_str(allocator: std.mem.Allocator, func: pr.Function) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (func.ops, 0..) |op, i| {
        if (i > 0) try w.writeByte('+');
        try w.writeAll(@tagName(op.prim()));
    }
    return try aw.toOwnedSlice();
}

fn build_providers_str(allocator: std.mem.Allocator, request: kernel.ProviderRequest) ![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    for (0..request.len()) |index| {
        if (index > 0) try output.writer.writeByte('|');
        try output.writer.writeAll(request.at(index));
    }
    return try output.toOwnedSlice();
}

fn build_shape_str(allocator: std.mem.Allocator, func: pr.Function) ![]const u8 {
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();
    const w = &aw.writer;
    for (func.params, 0..) |in_var, i| {
        if (i > 0) try w.writeByte('x');
        try kernel.write_aval_signature(w, in_var.aval);
    }
    return try aw.toOwnedSlice();
}

fn dump_kernel_entries(out: *std.Io.Writer, entries: []const KernelEntry) !void {
    var provider: usize = 0;
    var unreplaced: usize = 0;
    for (entries) |e| switch (e.outcome) {
        .provider => provider += 1,
        .unreplaced => unreplaced += 1,
    };
    try out.print("kernels: {d} provider, {d} unreplaced\n", .{ provider, unreplaced });
    try out.print("  {s:<50} {s:<10} {s:<30} {s:<50} {s}\n", .{ "function", "provider", "ops", "shapes", "outcome" });
    try out.writeAll("  " ++ (&@as([150]u8, @splat('-'))) ++ "\n");
    for (entries) |e| {
        try out.print("  {s:<50} {s:<10} {s:<30} {s:<50} {s}\n", .{
            e.name, e.provider, e.ops, e.shape, @tagName(e.outcome),
        });
    }
}

// Tests.

const test_device: device.Device = .{ .platform = .cpu };

fn unsupported_compile(
    _: *anyopaque,
    _: pr.Function,
    _: device.Device,
    _: std.mem.Allocator,
) kernel.CompileError!kernel.Artifact {
    return error.Unsupported;
}

const TestProvider = struct {
    name: []const u8,

    fn interface(self: *TestProvider) kernel.KernelProvider {
        return .{
            .name = self.name,
            .compiler = .{
                .context = @ptrCast(self),
                .vtable = &.{ .compile = unsupported_compile },
            },
        };
    }
};

fn make_test_selection_key(
    allocator: std.mem.Allocator,
    candidate: kernel.CandidateRegion,
    func: pr.Function,
) !kernel.SelectionKey {
    const function_fingerprint = try fingerprint.function(allocator, func);
    return try kernel.make_selection_key(
        allocator,
        candidate,
        test_device,
        function_fingerprint,
    );
}

fn select_provider_for_test(
    store: *kernel.KernelStore,
    selection_key: kernel.SelectionKey,
    provider_name: []const u8,
    data: []const u8,
) !void {
    try store.put(selection_key, .{
        .candidate = .{ .provider = .{
            .provider_name = provider_name,
            .artifact = .{ .data = try store.allocator().dupe(u8, data) },
        } },
        .reason = "available",
    });
}

fn extract_test_candidates(
    program: *pr.Program,
    provider_names: []const []const u8,
) !kernel.ExtractedCandidates {
    const states = try std.testing.allocator.alloc(TestProvider, provider_names.len);
    defer std.testing.allocator.free(states);
    const providers = try std.testing.allocator.alloc(kernel.KernelProvider, provider_names.len);
    defer std.testing.allocator.free(providers);
    for (provider_names, states, providers) |name, *state, *provider| {
        state.* = .{ .name = name };
        provider.* = state.interface();
    }

    var discovered = kernel.Candidates.init(std.testing.allocator);
    defer discovered.deinit();
    var extracted = kernel.ExtractedCandidates.init(std.testing.allocator);
    errdefer extracted.deinit();
    var ctx = compilation.Context{ .allocator = std.testing.allocator, .io = std.testing.io };
    _ = try (DiscoverCandidates{
        .providers = providers,
        .candidates = &discovered,
    }).run(program, &ctx);
    _ = try (ExtractCandidates{
        .candidates = &discovered,
        .extracted = &extracted,
    }).run(program, &ctx);
    return extracted;
}

test "ExtractCandidates retains nested explicit alternatives" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("outer", &.{kernel.provider_annotation("outer_provider")});
    const sum = try builder.add(lhs, rhs);
    try builder.push_region("inner", &.{kernel.provider_annotation("inner_provider")});
    const product = try builder.multiply(sum, rhs);
    try builder.pop_region();
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{product} }));

    var candidates = try extract_test_candidates(
        &program,
        &.{ "outer_provider", "inner_provider" },
    );
    defer candidates.deinit();

    try testing.expectEqual(@as(usize, 3), program.functions().len);
    try testing.expectEqual(pr.Prim.add, program.functions()[0].ops[0].prim());
    try testing.expectEqual(@as(usize, 2), candidates.entries.items.len);
    var found_outer = false;
    var found_inner = false;
    for (candidates.entries.items) |candidate| {
        const provider_name = candidate.boundary.request().at(0);
        found_outer = found_outer or std.mem.eql(u8, provider_name, "outer_provider");
        found_inner = found_inner or std.mem.eql(u8, provider_name, "inner_provider");
    }
    try testing.expect(found_outer);
    try testing.expect(found_inner);
    var extracted_region_count: usize = 0;
    for (candidates.entries.items) |candidate| {
        const callable = program.get_function_by_id(candidate.callable_function) orelse
            return error.TestUnexpectedResult;
        extracted_region_count += callable.regions.len;
        for (callable.regions) |region|
            try testing.expect((try kernel.requested_providers(region.annotations)) == null);
    }
    try testing.expectEqual(@as(usize, 1), extracted_region_count);
}

test "DiscoverCandidates groups providers matching the same PR range" {
    const testing = std.testing;
    const Matcher = struct {
        name: []const u8,

        fn provider(self: *@This()) kernel.KernelProvider {
            return .{
                .name = self.name,
                .compiler = .{
                    .context = @ptrCast(self),
                    .vtable = &.{
                        .compile = unsupported_compile,
                        .discover = discover,
                    },
                },
            };
        }

        fn discover(
            _: *anyopaque,
            func: pr.Function,
            matches: *std.ArrayList(kernel.Match),
            allocator: std.mem.Allocator,
        ) kernel.DiscoverError!void {
            for (func.ops, 0..) |op, op_index| {
                if (op.prim() != .mm) continue;
                try matches.append(allocator, .{ .start = op_index, .end = op_index + 1 });
            }
        }
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const first_lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const first_rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    const second_lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const second_rhs = try builder.param_tensor(.f32, &.{ 3, 2 });
    const first_result = try builder.mm(first_lhs, first_rhs);
    const second_result = try builder.mm(second_lhs, second_rhs);
    _ = try program.add_function(try builder.finish(.{
        .returns = &.{ first_result, second_result },
    }));

    var first = Matcher{ .name = "first" };
    var second = Matcher{ .name = "second" };
    var providers = [_]kernel.KernelProvider{ first.provider(), second.provider() };
    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (DiscoverCandidates{
        .providers = &providers,
        .candidates = &candidates,
    }).run(&program, &ctx);

    try testing.expectEqual(@as(usize, 0), program.functions()[0].regions.len);
    try testing.expectEqual(@as(usize, 2), candidates.entries.items.len);
    for (candidates.entries.items) |candidate| {
        const request = candidate.request();
        try testing.expectEqual(@as(usize, 2), request.len());
        try testing.expectEqualStrings("first", request.at(0));
        try testing.expectEqualStrings("second", request.at(1));
    }
}

test "DiscoverCandidates rejects an unconfigured request transactionally" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("configured", &.{kernel.provider_annotation("configured")});
    const first = try builder.exp(input);
    try builder.pop_region();
    try builder.push_region("missing", &.{kernel.provider_annotation("missing")});
    const output = try builder.log(first);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var provider_state = TestProvider{ .name = "configured" };
    const providers = [_]kernel.KernelProvider{provider_state.interface()};
    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    try testing.expectError(
        error.ProviderNotConfigured,
        (DiscoverCandidates{
            .providers = &providers,
            .candidates = &candidates,
        }).run(&program, &context),
    );
    try testing.expectEqual(@as(usize, 0), candidates.entries.items.len);
}

test "DiscoverCandidates retains opportunistic overlaps with explicit requests" {
    const testing = std.testing;
    const Matcher = struct {
        fn interface(self: *@This()) kernel.KernelProvider {
            return .{
                .name = "opportunistic",
                .compiler = .{
                    .context = @ptrCast(self),
                    .vtable = &.{
                        .compile = unsupported_compile,
                        .discover = discover,
                    },
                },
            };
        }

        fn discover(
            _: *anyopaque,
            _: pr.Function,
            matches: *std.ArrayList(kernel.Match),
            allocator: std.mem.Allocator,
        ) kernel.DiscoverError!void {
            try matches.append(allocator, .{ .start = 1, .end = 3 });
        }
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("declared", &.{kernel.provider_annotation("declared")});
    const first = try builder.exp(input);
    const second = try builder.log(first);
    try builder.pop_region();
    const output = try builder.add(second, input);
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var declared = TestProvider{ .name = "declared" };
    var matcher = Matcher{};
    const providers = [_]kernel.KernelProvider{ declared.interface(), matcher.interface() };
    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (DiscoverCandidates{
        .providers = &providers,
        .candidates = &candidates,
    }).run(&program, &context);

    try testing.expectEqual(@as(usize, 2), candidates.entries.items.len);
    try testing.expect(candidates.entries.items[0].explicit_region != null);
    try testing.expect(candidates.entries.items[1].explicit_region == null);
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, candidates.entries.items[0].op_ids);
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, candidates.entries.items[1].op_ids);
}

test "ExtractCandidates rejects a stale operation identity" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("candidate", &.{kernel.provider_annotation("provider")});
    const output = try builder.exp(input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var extracted = kernel.ExtractedCandidates.init(testing.allocator);
    defer extracted.deinit();
    var provider_state = TestProvider{ .name = "provider" };
    const providers = [_]kernel.KernelProvider{provider_state.interface()};
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (DiscoverCandidates{
        .providers = &providers,
        .candidates = &candidates,
    }).run(&program, &context);
    @constCast(candidates.entries.items[0].op_ids)[0] = std.math.maxInt(u32);

    try testing.expectError(
        error.StaleKernelCandidate,
        (ExtractCandidates{
            .candidates = &candidates,
            .extracted = &extracted,
        }).run(&program, &context),
    );
}

test "ExtractCandidates rolls back earlier extractions on failure" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const first = try builder.exp(input);
    const second = try builder.log(first);
    const source = try builder.finish(.{ .returns = &.{second} });
    const source_id = try program.add_function(source);

    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var extracted = kernel.ExtractedCandidates.init(testing.allocator);
    defer extracted.deinit();
    try candidates.add(source_id, source, .{ .start = 0, .end = 1 }, "provider", null);
    try candidates.add(source_id, source, .{ .start = 1, .end = 2 }, "provider", null);
    @constCast(candidates.entries.items[1].op_ids)[0] = std.math.maxInt(u32);

    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    try testing.expectError(
        error.StaleKernelCandidate,
        (ExtractCandidates{
            .candidates = &candidates,
            .extracted = &extracted,
        }).run(&program, &context),
    );
    try testing.expectEqual(@as(usize, 1), program.functions().len);
    try testing.expectEqual(@as(usize, 0), extracted.entries.items.len);
}

test "kernelization substitutes an opportunistic candidate without a PR region" {
    const testing = std.testing;
    const Matcher = struct {
        fn provider(self: *@This()) kernel.KernelProvider {
            return .{
                .name = "matcher",
                .compiler = .{
                    .context = @ptrCast(self),
                    .vtable = &.{
                        .compile = unsupported_compile,
                        .discover = discover,
                    },
                },
            };
        }

        fn discover(
            _: *anyopaque,
            func: pr.Function,
            matches: *std.ArrayList(kernel.Match),
            allocator: std.mem.Allocator,
        ) kernel.DiscoverError!void {
            for (func.ops, 0..) |op, op_index| {
                if (op.prim() != .exp) continue;
                try matches.append(allocator, .{ .start = op_index, .end = op_index + 1 });
            }
        }
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const output = try builder.exp(input);
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var matcher = Matcher{};
    const providers = [_]kernel.KernelProvider{matcher.provider()};
    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var extracted = kernel.ExtractedCandidates.init(testing.allocator);
    defer extracted.deinit();
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try (DiscoverCandidates{
        .providers = &providers,
        .candidates = &candidates,
    }).run(&program, &context);
    _ = try (ExtractCandidates{
        .candidates = &candidates,
        .extracted = &extracted,
    }).run(&program, &context);

    try testing.expectEqual(@as(usize, 0), program.functions()[0].regions.len);
    try testing.expectEqual(@as(usize, 1), candidates.entries.items.len);
    const candidate = extracted.entries.items[0];
    const callable = program.get_function_by_id(candidate.callable_function) orelse
        return error.TestUnexpectedResult;
    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidate.boundary,
        callable,
    );
    defer testing.allocator.free(selection_key.bytes);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try select_provider_for_test(&store, selection_key, "matcher", "payload");
    var kernelize = KernelizePass{
        .store = &store,
        .candidates = &extracted,
        .device = test_device,
    };
    _ = try kernelize.run(&program, &context);

    try testing.expectEqual(@as(usize, 0), program.functions()[0].regions.len);
    try testing.expectEqual(pr.Prim.custom_call, program.functions()[0].ops[0].prim());
}

test "DiscoverCandidates retains overlapping alternatives" {
    const testing = std.testing;
    const Matcher = struct {
        name: []const u8,
        range: pattern.Range,

        fn provider(self: *@This()) kernel.KernelProvider {
            return .{
                .name = self.name,
                .compiler = .{
                    .context = @ptrCast(self),
                    .vtable = &.{
                        .compile = unsupported_compile,
                        .discover = discover,
                    },
                },
            };
        }

        fn discover(
            ptr: *anyopaque,
            _: pr.Function,
            matches: *std.ArrayList(kernel.Match),
            allocator: std.mem.Allocator,
        ) kernel.DiscoverError!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            try matches.append(allocator, self.range);
        }
    };

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const first = try builder.exp(input);
    const second = try builder.log(first);
    const output = try builder.add(second, input);
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var left = Matcher{ .name = "left", .range = .{ .start = 0, .end = 2 } };
    var right = Matcher{ .name = "right", .range = .{ .start = 1, .end = 3 } };
    const providers = [_]kernel.KernelProvider{ left.provider(), right.provider() };
    var candidates = kernel.Candidates.init(testing.allocator);
    defer candidates.deinit();
    var extracted = kernel.ExtractedCandidates.init(testing.allocator);
    defer extracted.deinit();
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };

    _ = try (DiscoverCandidates{
        .providers = &providers,
        .candidates = &candidates,
    }).run(&program, &context);

    try testing.expectEqual(@as(usize, 2), candidates.entries.items.len);
    try testing.expectEqual(@as(usize, 1), program.functions().len);
    try testing.expectEqual(@as(usize, 3), program.functions()[0].ops.len);
    _ = try (ExtractCandidates{
        .candidates = &candidates,
        .extracted = &extracted,
    }).run(&program, &context);
    try testing.expectEqual(@as(usize, 3), program.functions().len);
    try testing.expectEqual(@as(usize, 3), program.functions()[0].ops.len);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    for (extracted.entries.items) |candidate| {
        const callable = program.get_function_by_id(candidate.callable_function) orelse
            return error.TestUnexpectedResult;
        const key = try make_test_selection_key(
            testing.allocator,
            candidate.boundary,
            callable,
        );
        defer testing.allocator.free(key.bytes);
        try select_provider_for_test(
            &store,
            key,
            candidate.boundary.request().at(0),
            "payload",
        );
    }
    var kernelize = KernelizePass{
        .store = &store,
        .candidates = &extracted,
        .device = test_device,
    };
    try testing.expectError(
        error.OverlappingKernelSelections,
        kernelize.run(&program, &context),
    );
    try testing.expectEqual(@as(usize, 3), program.functions()[0].ops.len);
}

test "kernelize pass rewrites a selected function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{y} });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidates.entries.items[0].boundary,
        program.functions()[1],
    );
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "stored_kernel_data");

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Region should be rewritten to custom_call.
    try testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);
    const rewritten = program.functions()[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());

    try testing.expectEqualStrings(kernel.dispatch_target_name, rewritten.params.custom_call.target_name);
    try testing.expectEqualStrings(selection_key.bytes, rewritten.params.custom_call.payload);
    try testing.expect(!rewritten.params.custom_call.has_side_effect);
}

test "kernelize pass preserves observable side effects" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "test");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("effectful", &.{kernel.provider_annotation("mock")});
    const outputs = (try builder.custom_call(.{
        .target_name = "test.effectful",
        .has_side_effect = true,
    }, &.{input}, &.{input.aval})).outputs;
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = outputs }));
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidates.entries.items[0].boundary,
        program.functions()[1],
    );
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

    var kernelize = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };
    var ctx = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try kernelize.run(&program, &ctx);

    const rewritten = program.functions()[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());
    try testing.expect(rewritten.params.custom_call.has_side_effect);
}

test "kernelize pass does not replace a declined function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{y} });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidates.entries.items[0].boundary,
        program.functions()[1],
    );
    defer testing.allocator.free(selection_key.bytes);
    try store.put(selection_key, .{
        .candidate = .unreplaced,
        .reason = "unsupported",
    });

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // An unreplaced candidate leaves the source operation intact.
    try testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);
    try testing.expectEqual(pr.Prim.exp, program.functions()[0].ops[0].prim());
}

test "kernelize pass does not replace a candidate without a selection" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", &.{kernel.provider_annotation("mock")});
    const y = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{y} });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    // An empty store leaves the source graph unchanged.
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // An unreplaced candidate leaves the source operation intact.
    try testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);
    try testing.expectEqual(pr.Prim.exp, program.functions()[0].ops[0].prim());
}

test "kernelize pass does not reuse another provider selection" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var builder = try pr.FunctionBuilder.init(&program, "test");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("test_region", &.{kernel.provider_annotation("requested")});
    const output = try builder.emit(.{ .exp = {} }, &.{input});
    try builder.pop_region();
    const function = try builder.finish(.{ .returns = &.{output} });
    _ = try program.add_function(function);
    var candidates = try extract_test_candidates(&program, &.{"requested"});
    defer candidates.deinit();

    var other_candidate = candidates.entries.items[0].boundary;
    other_candidate.provider_names = &.{"other"};
    const other_key = try make_test_selection_key(
        testing.allocator,
        other_candidate,
        program.functions()[1],
    );
    defer testing.allocator.free(other_key.bytes);
    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try select_provider_for_test(&store, other_key, "other", "payload");

    var kernelize = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };
    var context = compilation.Context{ .allocator = testing.allocator, .io = testing.io };
    _ = try kernelize.run(&program, &context);

    try testing.expectEqual(pr.Prim.exp, program.functions()[0].ops[0].prim());
}

test "kernelize pass rewrites a multi-output function call" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("multi_out", &.{kernel.provider_annotation("mock")});
    const a = try b.emit(.{ .exp = {} }, &.{x});
    const b_out = try b.emit(.{ .log = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{ a, b_out } });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidates.entries.items[0].boundary,
        program.functions()[1],
    );
    defer testing.allocator.free(selection_key.bytes);

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    try select_provider_for_test(&store, selection_key, "mock", "mock_kernel_data");

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    try testing.expectEqual(@as(usize, 1), program.functions()[0].ops.len);
    const rewritten = program.functions()[0].ops[0];
    try testing.expectEqual(pr.Prim.custom_call, rewritten.prim());

    try testing.expectEqual(@as(usize, 2), rewritten.outputs.len);

    try testing.expectEqual(x.aval, rewritten.outputs[0].aval);
    try testing.expectEqual(y.aval, rewritten.outputs[1].aval);
}

test "kernelize pass isolates equal callable occurrences" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // Both candidates have the same semantics.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("region_a", &.{kernel.provider_annotation("mock")});
    const out_a = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_b", &.{kernel.provider_annotation("mock")});
    const out_b = try b.emit(.{ .exp = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{ out_a, out_b } });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const selection_key = try make_test_selection_key(
        testing.allocator,
        candidates.entries.items[0].boundary,
        program.functions()[1],
    );
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // The selection belongs only to the first source occurrence.
    const ops = program.functions()[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.exp, ops[1].prim());
}

test "kernelize pass separates functions with different shapes" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    // The different shapes produce distinct kernel signatures.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{4});

    try b.push_region("region_small", &.{kernel.provider_annotation("mock")});
    const out_small = try b.emit(.{ .exp = {} }, &.{x});
    try b.pop_region();

    try b.push_region("region_large", &.{kernel.provider_annotation("mock")});
    const out_large = try b.emit(.{ .exp = {} }, &.{y});
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{ out_small, out_large } });
    _ = try program.add_function(func);
    var candidates = try extract_test_candidates(&program, &.{"mock"});
    defer candidates.deinit();

    var store = kernel.KernelStore.init(testing.allocator);
    defer store.deinit();
    const small_candidate = for (candidates.entries.items) |candidate| {
        if (candidate.boundary.op_ids[0] == program.functions()[0].ops[0].id)
            break candidate;
    } else return error.TestUnexpectedResult;
    const small_func = program.get_function_by_id(small_candidate.callable_function) orelse
        return error.TestUnexpectedResult;
    const selection_key = try make_test_selection_key(
        testing.allocator,
        small_candidate.boundary,
        small_func,
    );
    defer testing.allocator.free(selection_key.bytes);
    try select_provider_for_test(&store, selection_key, "mock", "payload");

    var kp = KernelizePass{
        .store = &store,
        .candidates = &candidates,
        .device = test_device,
    };

    var ctx = compilation.Context{ .allocator = testing.allocator, .io = std.testing.io };
    _ = try kp.run(&program, &ctx);

    // Only `region_small` has a store selection. `region_large` is a call.
    const ops = program.functions()[0].ops;
    try testing.expectEqual(@as(usize, 2), ops.len);
    try testing.expectEqual(pr.Prim.custom_call, ops[0].prim());
    try testing.expectEqual(pr.Prim.exp, ops[1].prim());
}
