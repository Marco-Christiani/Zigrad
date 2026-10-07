//! Kernel candidate compilation, measurement, and resolution.
//!
//! Candidate collection retains every compiled implementation and its
//!  measurement. A resolver sees the complete collection before final
//!  selections enter a `KernelStore`.
//!
//! ```zig
//! const zg = @import("zigrad");
//!
//! var store = try zg.tune.tune(io, allocator, &program, providers, .{
//!     .device = target,
//!     .candidates = &extracted,
//!     .measurer = measurer,
//! });
//! defer store.deinit();
//!
//! var kernelize = zg.pr.transform.kernelize.KernelizePass{
//!     .store = &store,
//!     .candidates = &extracted,
//!     .device = target,
//! };
//! _ = try kernelize.run(&program, &context);
//! ```

const std = @import("std");
const device = @import("device");
const fingerprint = @import("pr").analysis.fingerprint;
const pr = @import("pr");
const kernel = @import("kernel");
const measurement = @import("tune/measurer.zig");

const log = std.log.scoped(.@"zg/tune");

pub const Measurement = measurement.Measurement;
pub const Sample = measurement.Sample;
pub const MeasurementError = measurement.Error;
pub const Measurer = measurement.Measurer;
pub const Executable = measurement.Executable;
pub const ExecutableFactory = measurement.ExecutableFactory;
pub const ExecutableMeasurer = measurement.ExecutableMeasurer;
pub const InputGenerator = measurement.InputGenerator;

/// Compiled implementations and measurements for one candidate boundary.
pub const EvaluatedCandidate = struct {
    /// Candidate boundary whose slices are borrowed from the extracted collection.
    boundary: kernel.CandidateRegion,
    /// Extracted callable borrowed from the PR program.
    callable: pr.Function,
    /// Provider artifacts owned by the enclosing `EvaluatedCandidates`.
    ///
    /// Provider-name bytes remain borrowed from `boundary`.
    implementations: []kernel.ProviderCandidate,
    /// Measurements aligned with `implementations`.
    measurements: []Measurement,

    fn deinit(self: *EvaluatedCandidate, allocator: std.mem.Allocator) void {
        for (self.implementations) |*implementation|
            implementation.artifact.deinit(allocator);
        allocator.free(self.implementations);
        measurement.deinit_measurements(allocator, self.measurements);
        self.* = undefined;
    }
};

/// Complete provider evidence retained until an explicit resolution step.
pub const EvaluatedCandidates = struct {
    /// Allocator owning entries, artifacts, and measurements.
    allocator: std.mem.Allocator,
    /// Evaluated candidates in extracted-candidate order.
    entries: []EvaluatedCandidate,

    /// Release compiled artifacts, measurements, and entry storage.
    pub fn deinit(self: *EvaluatedCandidates) void {
        for (self.entries) |*entry| entry.deinit(self.allocator);
        self.allocator.free(self.entries);
        self.* = undefined;
    }
};

/// Options for compiling and measuring discovered candidates.
pub const CollectOpts = struct {
    /// Device passed to providers and the optional measurer.
    device: device.Device,
    /// Extracted candidates to compile and measure.
    candidates: *const kernel.ExtractedCandidates,
    /// Measurement mechanism. Missing measurement leaves every implementation
    ///  available but without profitability evidence.
    measurer: ?Measurer = null,
};

/// Compile every eligible provider implementation and retain all measurements.
///
/// Every candidate must first pass through `pr.transform.kernelize.ExtractCandidates`.
///
/// Returned result borrows candidate and PR storage.
pub fn collect(
    /// Allocator owning compiled artifacts, measurements, and result storage.
    allocator: std.mem.Allocator,
    /// Program containing every extracted callable.
    program: *const pr.Program,
    /// Providers eligible to compile discovered candidates.
    providers: []const kernel.KernelProvider,
    /// Target, candidate collection, and optional measurement mechanism.
    opts: CollectOpts,
) !EvaluatedCandidates {
    try kernel.validate_providers(providers);

    var entries = std.ArrayList(EvaluatedCandidate).empty;
    errdefer deinit_entry_list(allocator, &entries);

    for (opts.candidates.entries.items) |candidate| {
        const boundary = candidate.boundary;
        const callable = program.get_function_by_id(candidate.callable_function) orelse
            return error.CallUnresolvedCallee;

        var implementations = std.ArrayList(kernel.ProviderCandidate).empty;
        errdefer {
            for (implementations.items) |implementation|
                allocator.free(implementation.artifact.data);
            implementations.deinit(allocator);
        }
        const request = boundary.request();
        for (0..request.len()) |request_index| {
            const provider_name = request.at(request_index);
            const provider = kernel.find_provider(providers, provider_name) orelse {
                log.info("candidate for function '{s}' requests unconfigured provider '{s}'", .{
                    callable.name,
                    provider_name,
                });
                return error.ProviderNotConfigured;
            };
            try implementations.ensureUnusedCapacity(allocator, 1);
            const artifact = provider.compiler.compile(
                callable,
                opts.device,
                allocator,
            ) catch |err| switch (err) {
                error.Unsupported => {
                    log.debug("provider '{s}' cannot handle function '{s}'", .{
                        provider_name,
                        callable.name,
                    });
                    continue;
                },
                else => {
                    log.err("provider '{s}' failed to compile function '{s}': {s}", .{
                        provider_name,
                        callable.name,
                        @errorName(err),
                    });
                    return err;
                },
            };
            implementations.appendAssumeCapacity(.{
                .provider_name = provider_name,
                .artifact = artifact,
            });
        }

        const owned_implementations = try implementations.toOwnedSlice(allocator);
        errdefer {
            for (owned_implementations) |implementation|
                allocator.free(implementation.artifact.data);
            allocator.free(owned_implementations);
        }
        const measurements = if (owned_implementations.len == 0) blk: {
            break :blk try allocator.alloc(Measurement, 0);
        } else if (opts.measurer) |measurer| blk: {
            const measured = try measurer.measure(
                callable,
                owned_implementations,
                opts.device,
                allocator,
            );
            if (measured.len != owned_implementations.len) {
                measurement.deinit_measurements(allocator, measured);
                return error.InvalidMeasurementSet;
            }
            if (!valid_measurements(measured)) {
                measurement.deinit_measurements(allocator, measured);
                return error.InvalidMeasurementSet;
            }
            break :blk measured;
        } else blk: {
            const unavailable = try allocator.alloc(Measurement, owned_implementations.len);
            @memset(unavailable, .unavailable);
            break :blk unavailable;
        };
        errdefer measurement.deinit_measurements(allocator, measurements);

        try entries.append(allocator, .{
            .boundary = boundary,
            .callable = callable,
            .implementations = owned_implementations,
            .measurements = measurements,
        });
    }

    return .{
        .allocator = allocator,
        .entries = try entries.toOwnedSlice(allocator),
    };
}

fn valid_measurements(measurements: []const Measurement) bool {
    for (measurements) |result| switch (result) {
        .unavailable, .incorrect => {},
        .measured => |samples| {
            if (samples.len == 0) return false;
            for (samples) |sample| {
                if (sample.unreplaced_ns == 0 or sample.selected_ns == 0)
                    return false;
            }
        },
    };
    return true;
}

fn deinit_entry_list(
    allocator: std.mem.Allocator,
    entries: *std.ArrayList(EvaluatedCandidate),
) void {
    for (entries.items) |*entry| entry.deinit(allocator);
    entries.deinit(allocator);
}

/// Final disposition of a candidate.
pub const Decision = struct {
    /// Source boundary identified by function and operation ids.
    boundary: kernel.CandidateRegion,
    /// Implementation selected for the boundary.
    implementation: union(enum) {
        /// Leave the source boundary available to the enclosing compiler.
        unreplaced,
        /// Select the implementation produced by this provider.
        provider: []const u8,
    },
    /// Explanation copied into the final kernel store.
    reason: []const u8,
    /// Measurement summary retained by the resolving policy.
    measurement: ?kernel.MeasurementEvidence = null,
};

/// Resolver output identified independently of candidate storage order.
///
/// The resolution owns the decision slice. Nested boundary, provider, and
///  reason slices are borrowed until the result is consumed.
pub const Resolution = struct {
    /// Allocator owning `decisions`.
    allocator: std.mem.Allocator,
    /// One decision for each evaluated boundary, in any order.
    decisions: []Decision,

    /// Release resolver-owned decision storage.
    pub fn deinit(self: *Resolution) void {
        self.allocator.free(self.decisions);
        self.* = undefined;
    }
};

/// Failures exposed by a type-erased resolution policy.
pub const ResolveError = std.mem.Allocator.Error || error{
    /// Resolver input or output violates the resolution contract.
    InvalidResolution,
    /// Resolver policy failed after emitting integration-specific diagnostics.
    ResolutionFailed,
    /// A candidate boundary no longer resolves in its source function.
    StaleKernelCandidate,
};

/// Type-erased policy over complete candidate evidence.
pub const Resolver = struct {
    /// Policy state borrowed for the lifetime of this interface.
    context: *anyopaque,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        resolve: *const fn (
            context: *anyopaque,
            program: *const pr.Program,
            evaluated: []const EvaluatedCandidate,
            allocator: std.mem.Allocator,
        ) ResolveError!Resolution,
    };

    /// Resolve provider and overlap choices for the complete collection.
    pub fn resolve(
        self: Resolver,
        /// Program containing the source boundaries under resolution.
        program: *const pr.Program,
        /// Candidate boundaries with aligned implementations and measurements.
        evaluated: []const EvaluatedCandidate,
        /// Allocator owning the returned decision slice.
        allocator: std.mem.Allocator,
    ) ResolveError!Resolution {
        return try self.vtable.resolve(self.context, program, evaluated, allocator);
    }
};

/// Local additive profitability policy with per-function interval resolution.
///
/// The policy first retains the statistically accepted implementation with the
///  largest estimated latency savings at each boundary. It then maximizes the
///  sum of those savings among non-overlapping boundaries in each source
///  function with weighted interval scheduling. Profitability uses a one-sided
///  paired sign test with a Bonferroni correction across implementations. *The
///  additive model does not represent cross-boundary runtime interactions.*
pub const LocalResolver = struct {
    pub const Options = struct {
        /// Family-wise one-sided sign-test threshold for each boundary.
        ///
        /// Bonferroni correction divides this value by the number of
        ///  implementations at that boundary.
        alpha: f64 = 0.05,
        /// Smallest accepted fractional reduction in median latency.
        minimum_speedup: f64 = 0.01,
    };

    /// Statistical thresholds used by the local policy.
    options: Options = .{},

    /// Return the type-erased resolver interface.
    pub fn interface(self: *LocalResolver) Resolver {
        return .{
            .context = @ptrCast(self),
            .vtable = &.{ .resolve = resolve_impl },
        };
    }

    fn resolve_impl(
        ptr: *anyopaque,
        program: *const pr.Program,
        evaluated: []const EvaluatedCandidate,
        allocator: std.mem.Allocator,
    ) ResolveError!Resolution {
        const self: *LocalResolver = @ptrCast(@alignCast(ptr));
        if (self.options.alpha <= 0.0 or self.options.alpha >= 1.0 or
            self.options.minimum_speedup < 0.0 or self.options.minimum_speedup >= 1.0)
        {
            return error.InvalidResolution;
        }

        const decisions = try allocator.alloc(Decision, evaluated.len);
        errdefer allocator.free(decisions);
        for (evaluated, decisions) |candidate, *decision| {
            decision.* = .{
                .boundary = candidate.boundary,
                .implementation = .unreplaced,
                .reason = if (candidate.implementations.len == 0)
                    "no requested provider produced an artifact"
                else
                    "no provider passed local profitability policy",
            };
        }

        var viable = std.ArrayList(ViableChoice).empty;
        defer viable.deinit(allocator);
        for (evaluated, 0..) |*candidate, candidate_index| {
            if (candidate.implementations.len != candidate.measurements.len)
                return error.InvalidResolution;
            const source = program.get_function_by_id(candidate.boundary.source_function) orelse
                return error.InvalidResolution;
            const range = candidate.boundary.resolve_range(source) orelse
                return error.StaleKernelCandidate;
            const selected = try best_local_implementation(
                allocator,
                candidate,
                self.options,
            ) orelse
                continue;
            try viable.append(allocator, .{
                .candidate_index = candidate_index,
                .provider_name = candidate.implementations[selected.index].provider_name,
                .source_function = candidate.boundary.source_function,
                .start = range.start,
                .end = range.end,
                .evidence = selected.evidence,
            });
            decisions[candidate_index].reason = "profitable candidate excluded by overlap resolution";
        }

        std.mem.sortUnstable(ViableChoice, viable.items, {}, viable_less_than);
        var start: usize = 0;
        while (start < viable.items.len) {
            var end = start + 1;
            while (end < viable.items.len and
                viable.items[end].source_function == viable.items[start].source_function)
            {
                end += 1;
            }
            try select_nonoverlapping(allocator, viable.items[start..end], decisions);
            start = end;
        }

        return .{ .allocator = allocator, .decisions = decisions };
    }
};

const ViableChoice = struct {
    candidate_index: usize,
    provider_name: []const u8,
    source_function: pr.FunctionId,
    start: usize,
    end: usize,
    evidence: kernel.MeasurementEvidence,
};

const LocalImplementation = struct {
    index: usize,
    evidence: kernel.MeasurementEvidence,
};

fn best_local_implementation(
    allocator: std.mem.Allocator,
    candidate: *const EvaluatedCandidate,
    options: LocalResolver.Options,
) ResolveError!?LocalImplementation {
    if (candidate.implementations.len == 0) return null;
    const adjusted_alpha = options.alpha /
        @as(f64, @floatFromInt(candidate.implementations.len));
    var best: ?LocalImplementation = null;
    for (candidate.measurements, 0..) |result, implementation_index| {
        const samples = switch (result) {
            .measured => |value| value,
            else => continue,
        };
        const evidence = try summarize_measurement(allocator, samples);
        const baseline = evidence.unreplaced_ns;
        const selected = evidence.selected_ns;
        if (selected >= baseline or evidence.p_value > adjusted_alpha)
            continue;
        const speedup = 1.0 - @as(f64, @floatFromInt(selected)) /
            @as(f64, @floatFromInt(baseline));
        if (speedup < options.minimum_speedup) continue;
        if (best) |incumbent| {
            const savings = baseline - selected;
            const incumbent_savings = incumbent.evidence.unreplaced_ns -
                incumbent.evidence.selected_ns;
            if (savings <= incumbent_savings) continue;
        }
        best = .{ .index = implementation_index, .evidence = evidence };
    }
    return best;
}

fn summarize_measurement(
    allocator: std.mem.Allocator,
    samples: []const Sample,
) ResolveError!kernel.MeasurementEvidence {
    if (samples.len == 0 or samples.len > 63) return error.InvalidResolution;
    const unreplaced = try allocator.alloc(u64, samples.len);
    defer allocator.free(unreplaced);
    const selected = try allocator.alloc(u64, samples.len);
    defer allocator.free(selected);
    for (samples, unreplaced, selected) |sample, *baseline, *candidate| {
        if (sample.unreplaced_ns == 0 or sample.selected_ns == 0)
            return error.InvalidResolution;
        baseline.* = sample.unreplaced_ns;
        candidate.* = sample.selected_ns;
    }
    return .{
        .unreplaced_ns = median(unreplaced),
        .selected_ns = median(selected),
        .p_value = paired_sign_test(samples),
    };
}

fn median(values: []u64) u64 {
    std.debug.assert(values.len > 0);
    std.mem.sort(u64, values, {}, std.sort.asc(u64));
    const middle = values.len / 2;
    if (values.len % 2 == 1) return values[middle];
    return values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

fn paired_sign_test(samples: []const Sample) f64 {
    std.debug.assert(samples.len <= 63);
    var wins: usize = 0;
    var observations: usize = 0;
    for (samples) |sample| {
        if (sample.unreplaced_ns == sample.selected_ns) continue;
        observations += 1;
        if (sample.selected_ns < sample.unreplaced_ns) wins += 1;
    }
    if (observations == 0) return 1.0;

    var combination: u128 = 1;
    var numerator: u128 = 0;
    for (0..observations + 1) |successes| {
        if (successes >= wins) numerator += combination;
        if (successes < observations) {
            combination = combination * (observations - successes) / (successes + 1);
        }
    }
    const denominator: u128 = @as(u128, 1) << @intCast(observations);
    return @as(f64, @floatFromInt(numerator)) /
        @as(f64, @floatFromInt(denominator));
}

fn viable_less_than(_: void, lhs: ViableChoice, rhs: ViableChoice) bool {
    if (lhs.source_function != rhs.source_function)
        return @backingInt(lhs.source_function) < @backingInt(rhs.source_function);
    if (lhs.end != rhs.end) return lhs.end < rhs.end;
    if (lhs.start != rhs.start) return lhs.start < rhs.start;
    return lhs.candidate_index < rhs.candidate_index;
}

fn select_nonoverlapping(
    allocator: std.mem.Allocator,
    candidates: []const ViableChoice,
    decisions: []Decision,
) std.mem.Allocator.Error!void {
    const best_savings = try allocator.alloc(u128, candidates.len + 1);
    defer allocator.free(best_savings);
    const compatible_prefix = try allocator.alloc(usize, candidates.len);
    defer allocator.free(compatible_prefix);
    const take = try allocator.alloc(bool, candidates.len);
    defer allocator.free(take);

    best_savings[0] = 0;
    for (candidates, 0..) |candidate, index| {
        var prefix = index;
        while (prefix > 0 and candidates[prefix - 1].end > candidate.start)
            prefix -= 1;
        compatible_prefix[index] = prefix;
        const with_candidate = best_savings[prefix] +
            candidate.evidence.unreplaced_ns - candidate.evidence.selected_ns;
        const without_candidate = best_savings[index];
        take[index] = with_candidate > without_candidate;
        best_savings[index + 1] = if (take[index]) with_candidate else without_candidate;
    }

    var cursor = candidates.len;
    while (cursor > 0) {
        const index = cursor - 1;
        if (!take[index]) {
            cursor = index;
            continue;
        }
        const candidate = candidates[index];
        decisions[candidate.candidate_index] = .{
            .boundary = decisions[candidate.candidate_index].boundary,
            .implementation = .{ .provider = candidate.provider_name },
            .reason = "provider passed local profitability and overlap policy",
            .measurement = candidate.evidence,
        };
        cursor = compatible_prefix[index];
    }
}

/// Materialize one resolver result as the final in-memory kernel plan.
///
/// Selected artifacts are copied so the evaluated collection is usable
///  by another resolver.
pub fn resolve(
    /// Allocator owning the returned kernel store.
    allocator: std.mem.Allocator,
    /// Program containing source boundaries and extracted callables.
    program: *const pr.Program,
    /// Complete candidate evidence passed to `resolver`.
    evaluated: *const EvaluatedCandidates,
    /// Device encoded into final selection keys.
    selected_device: device.Device,
    /// Policy that selects implementations and resolves conflicts.
    resolver: Resolver,
) !kernel.KernelStore {
    var resolution = try resolver.resolve(program, evaluated.entries, allocator);
    defer resolution.deinit();
    if (resolution.decisions.len != evaluated.entries.len)
        return error.InvalidResolution;

    var store = kernel.KernelStore.init(allocator);
    errdefer store.deinit();
    const resolved = try allocator.alloc(bool, evaluated.entries.len);
    defer allocator.free(resolved);
    @memset(resolved, false);
    for (resolution.decisions) |decision| {
        if (decision.measurement) |evidence| {
            if (!valid_measurement_evidence(evidence))
                return error.InvalidResolution;
        }
        const candidate_index = find_evaluated_candidate(evaluated.entries, decision.boundary) orelse
            return error.InvalidResolution;
        if (resolved[candidate_index]) return error.InvalidResolution;
        resolved[candidate_index] = true;
        const candidate = evaluated.entries[candidate_index];
        const function_fingerprint = try fingerprint.function(allocator, candidate.callable);
        const key = try kernel.make_selection_key(
            allocator,
            candidate.boundary,
            selected_device,
            function_fingerprint,
        );
        defer allocator.free(key.bytes);

        switch (decision.implementation) {
            .unreplaced => try store.put(key, .{
                .candidate = .unreplaced,
                .reason = decision.reason,
                .measurement = decision.measurement,
            }),
            .provider => |provider_name| {
                const implementation_index = find_implementation(
                    candidate.implementations,
                    provider_name,
                ) orelse return error.InvalidResolution;
                const selected = candidate.implementations[implementation_index];
                try store.put(key, .{
                    .candidate = .{ .provider = .{
                        .provider_name = selected.provider_name,
                        .artifact = .{
                            .data = try allocator.dupe(u8, selected.artifact.data),
                            .workspace_bytes = selected.artifact.workspace_bytes,
                            .workspace_alignment = selected.artifact.workspace_alignment,
                        },
                    } },
                    .reason = decision.reason,
                    .measurement = decision.measurement,
                });
            },
        }
    }
    for (resolved) |found| if (!found) return error.InvalidResolution;
    return store;
}

fn valid_measurement_evidence(evidence: kernel.MeasurementEvidence) bool {
    return evidence.unreplaced_ns > 0 and evidence.selected_ns > 0 and
        std.math.isFinite(evidence.p_value) and
        evidence.p_value >= 0.0 and evidence.p_value <= 1.0;
}

fn find_evaluated_candidate(
    evaluated: []const EvaluatedCandidate,
    boundary: kernel.CandidateRegion,
) ?usize {
    for (evaluated, 0..) |candidate, index| {
        if (candidate.boundary.same_boundary(boundary)) return index;
    }
    return null;
}

fn find_implementation(
    implementations: []const kernel.ProviderCandidate,
    provider_name: []const u8,
) ?usize {
    for (implementations, 0..) |implementation, index| {
        if (std.mem.eql(u8, implementation.provider_name, provider_name)) return index;
    }
    return null;
}

/// Options for the compile, measure, and resolve convenience entrypoint.
pub const TuneOpts = struct {
    /// Print a summary table after resolution.
    dump_results: bool = false,
    /// Device passed through collection and final selection identity.
    device: device.Device,
    /// Extracted candidates to compile and measure.
    candidates: *const kernel.ExtractedCandidates,
    /// Optional target measurement mechanism.
    measurer: ?Measurer = null,
    /// Optional policy over the complete evaluated collection.
    ///
    /// The local additive policy is used when null.
    resolver: ?Resolver = null,
};

/// Compile, measure, resolve, and return a final in-memory kernel plan.
pub fn tune(
    /// I/O state used for timing and diagnostics when applicable.
    io: std.Io,
    /// Allocator owning temporary evidence and the returned kernel store.
    allocator: std.mem.Allocator,
    /// Program containing every extracted candidate callable.
    program: *const pr.Program,
    /// Providers eligible to compile discovered candidates.
    providers: []const kernel.KernelProvider,
    /// Target, candidates, measurement, and resolution policy.
    opts: TuneOpts,
) !kernel.KernelStore {
    const start = std.Io.Timestamp.now(io, .awake);
    var evaluated = try collect(allocator, program, providers, .{
        .device = opts.device,
        .candidates = opts.candidates,
        .measurer = opts.measurer,
    });
    defer evaluated.deinit();

    var local_resolver = LocalResolver{};
    var store = try resolve(
        allocator,
        program,
        &evaluated,
        opts.device,
        opts.resolver orelse local_resolver.interface(),
    );
    errdefer store.deinit();

    log.info("tuning resolved {d} candidate boundaries in {d:.2}ms", .{
        evaluated.entries.len,
        ns_to_ms(@intCast(start.untilNow(io, .awake).toNanoseconds())),
    });
    if (opts.dump_results) dump_store_summary(io, &store);
    return store;
}

fn dump_store_summary(io: std.Io, store: *const kernel.KernelStore) void {
    var buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &buf);
    const out = &writer.interface;
    out.writeAll("\n=== Tuning Summary ===\n") catch return;
    var iterator = store.selections.iterator();
    while (iterator.next()) |entry| {
        const selection = entry.value_ptr.*;
        switch (selection.candidate) {
            .provider => |stored| {
                out.print("  [+] {s}: provider={s}, bytes={d}, ws={d}", .{
                    entry.key_ptr.*,
                    stored.provider_name,
                    stored.artifact.data.len,
                    stored.artifact.workspace_bytes,
                }) catch return;
                dump_measurement(out, selection.measurement);
            },
            .unreplaced => {
                out.print("  [=] {s}: unreplaced, reason={s}", .{
                    entry.key_ptr.*,
                    selection.reason,
                }) catch return;
                dump_measurement(out, selection.measurement);
            },
        }
    }
    out.writeByte('\n') catch {};
    out.flush() catch {};
}

fn dump_measurement(out: *std.Io.Writer, measurement_evidence: ?kernel.MeasurementEvidence) void {
    if (measurement_evidence) |measured| {
        out.print(", unreplaced={d}ns, selected={d}ns, p={d:.4}", .{
            measured.unreplaced_ns,
            measured.selected_ns,
            measured.p_value,
        }) catch return;
    }
    out.writeByte('\n') catch {};
}

fn ns_to_ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

// Tests.

const TestProvider = struct {
    name: []const u8,
    unsupported: bool = false,
    calls: usize = 0,

    fn interface(self: *TestProvider) kernel.KernelProvider {
        return .{
            .name = self.name,
            .compiler = .{
                .context = @ptrCast(self),
                .vtable = &.{ .compile = compile },
            },
        };
    }

    fn compile(
        ptr: *anyopaque,
        _: pr.Function,
        _: device.Device,
        allocator: std.mem.Allocator,
    ) kernel.CompileError!kernel.Artifact {
        const self: *TestProvider = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        if (self.unsupported) return error.Unsupported;
        return .{ .data = try allocator.dupe(u8, self.name) };
    }
};

const TestMeasurer = struct {
    calls: usize = 0,

    fn interface(self: *TestMeasurer) Measurer {
        return .{
            .context = @ptrCast(self),
            .vtable = &.{ .measure = measure },
        };
    }

    fn measure(
        ptr: *anyopaque,
        _: pr.Function,
        candidates: []const kernel.ProviderCandidate,
        _: device.Device,
        allocator: std.mem.Allocator,
    ) MeasurementError![]Measurement {
        const self: *TestMeasurer = @ptrCast(@alignCast(ptr));
        self.calls += 1;
        const results = try allocator.alloc(Measurement, candidates.len);
        var result_count: usize = 0;
        errdefer {
            for (results[0..result_count]) |result| result.deinit(allocator);
            allocator.free(results);
        }
        for (candidates, results, 0..) |_, *result, index| {
            const samples = try allocator.alloc(Sample, 15);
            @memset(samples, .{
                .unreplaced_ns = 100,
                .selected_ns = 80 - @as(u64, @intCast(index * 10)),
            });
            result.* = .{ .measured = samples };
            result_count += 1;
        }
        return results;
    }
};

fn discover_and_extract(
    program: *pr.Program,
    providers: []const kernel.KernelProvider,
) !kernel.ExtractedCandidates {
    const kernelize = @import("pr").transform.kernelize;
    var discovered = kernel.Candidates.init(std.testing.allocator);
    defer discovered.deinit();
    var extracted = kernel.ExtractedCandidates.init(std.testing.allocator);
    errdefer extracted.deinit();
    var context = @import("compilation").Context{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    _ = try (kernelize.DiscoverCandidates{
        .providers = providers,
        .candidates = &discovered,
    }).run(program, &context);
    _ = try (kernelize.ExtractCandidates{
        .candidates = &discovered,
        .extracted = &extracted,
    }).run(program, &context);
    return extracted;
}

test "collect retains every compiled implementation and measurement" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("candidate", &.{kernel.providers_annotation(&.{ "first", "second" })});
    const output = try builder.exp(input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{output} }));

    var first = TestProvider{ .name = "first" };
    var second = TestProvider{ .name = "second" };
    const providers = [_]kernel.KernelProvider{ first.interface(), second.interface() };
    var candidates = try discover_and_extract(&program, &providers);
    defer candidates.deinit();
    var measurer = TestMeasurer{};
    var evaluated = try collect(testing.allocator, &program, &providers, .{
        .device = .{ .platform = .cpu },
        .candidates = &candidates,
        .measurer = measurer.interface(),
    });
    defer evaluated.deinit();

    try testing.expectEqual(@as(usize, 1), evaluated.entries.len);
    try testing.expectEqual(@as(usize, 2), evaluated.entries[0].implementations.len);
    try testing.expectEqual(@as(usize, 2), evaluated.entries[0].measurements.len);
    for (evaluated.entries[0].measurements) |result| switch (result) {
        .measured => |samples| try testing.expectEqual(@as(usize, 15), samples.len),
        else => return error.TestUnexpectedResult,
    };
    try testing.expectEqual(@as(usize, 1), measurer.calls);
}

test "collect rejects a missing requested provider" {
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

    var provider = TestProvider{ .name = "provider" };
    const discovery_providers = [_]kernel.KernelProvider{provider.interface()};
    var candidates = try discover_and_extract(&program, &discovery_providers);
    defer candidates.deinit();

    try testing.expectError(
        error.ProviderNotConfigured,
        collect(testing.allocator, &program, &.{}, .{
            .device = .{ .platform = .cpu },
            .candidates = &candidates,
        }),
    );
}

test "local measurement summary retains paired-test semantics" {
    var samples = @as([10]Sample, @splat(.{ .unreplaced_ns = 100, .selected_ns = 90 }));
    const evidence = try summarize_measurement(std.testing.allocator, &samples);
    try std.testing.expectEqual(@as(u64, 100), evidence.unreplaced_ns);
    try std.testing.expectEqual(@as(u64, 90), evidence.selected_ns);
    try std.testing.expectApproxEqAbs(
        @as(f64, 1.0 / 1024.0),
        evidence.p_value,
        1e-12,
    );
}

test "tune uses occurrence-specific selection identity" {
    const testing = std.testing;
    const selected_device = device.Device{ .platform = .cpu };
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const first_input = try builder.param_tensor(.f32, &.{2});
    const second_input = try builder.param_tensor(.f32, &.{2});
    try builder.push_region("first", &.{kernel.provider_annotation("provider")});
    const first = try builder.exp(first_input);
    try builder.pop_region();
    try builder.push_region("second", &.{kernel.provider_annotation("provider")});
    const second = try builder.exp(second_input);
    try builder.pop_region();
    _ = try program.add_function(try builder.finish(.{ .returns = &.{ first, second } }));

    var provider = TestProvider{ .name = "provider" };
    const providers = [_]kernel.KernelProvider{provider.interface()};
    var candidates = try discover_and_extract(&program, &providers);
    defer candidates.deinit();
    var measurer = TestMeasurer{};
    var store = try tune(testing.io, testing.allocator, &program, &providers, .{
        .device = selected_device,
        .candidates = &candidates,
        .measurer = measurer.interface(),
    });
    defer store.deinit();

    try testing.expectEqual(@as(usize, 2), store.selections.count());
    try testing.expectEqual(@as(usize, 2), provider.calls);
    try testing.expectEqual(@as(usize, 2), measurer.calls);
}

test "local resolver leaves unmeasured implementations unreplaced" {
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

    var provider = TestProvider{ .name = "provider" };
    const providers = [_]kernel.KernelProvider{provider.interface()};
    var candidates = try discover_and_extract(&program, &providers);
    defer candidates.deinit();
    var store = try tune(testing.io, testing.allocator, &program, &providers, .{
        .device = .{ .platform = .cpu },
        .candidates = &candidates,
    });
    defer store.deinit();
    try testing.expectEqual(@as(usize, 1), store.selections.count());
    var iterator = store.selections.valueIterator();
    const selection = iterator.next() orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .unreplaced => {},
        .provider => return error.TestUnexpectedResult,
    }
}

test "local resolver maximizes nonoverlapping estimated savings" {
    const testing = std.testing;
    var program = pr.Program.init(testing.allocator);
    defer program.deinit();
    var builder = try pr.FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const input = try builder.param_tensor(.f32, &.{2});
    const first = try builder.exp(input);
    const second = try builder.log(first);
    const third = try builder.exp(second);
    const source = try builder.finish(.{ .returns = &.{third} });
    const source_id = try program.add_function(source);

    const provider_names = [_][]const u8{"provider"};
    const wide_ops = [_]u32{ source.ops[0].id, source.ops[1].id };
    const left_ops = [_]u32{source.ops[0].id};
    const right_ops = [_]u32{source.ops[1].id};
    var implementations = [_][1]kernel.ProviderCandidate{
        .{.{ .provider_name = "provider", .artifact = .{ .data = @constCast("wide") } }},
        .{.{ .provider_name = "provider", .artifact = .{ .data = @constCast("left") } }},
        .{.{ .provider_name = "provider", .artifact = .{ .data = @constCast("right") } }},
    };
    var wide_samples = @as([15]Sample, @splat(.{ .unreplaced_ns = 100, .selected_ns = 60 }));
    var left_samples = @as([15]Sample, @splat(.{ .unreplaced_ns = 100, .selected_ns = 70 }));
    var right_samples = @as([15]Sample, @splat(.{ .unreplaced_ns = 100, .selected_ns = 70 }));
    var measurements = [_][1]Measurement{
        .{.{ .measured = &wide_samples }},
        .{.{ .measured = &left_samples }},
        .{.{ .measured = &right_samples }},
    };
    const evaluated = [_]EvaluatedCandidate{
        .{
            .boundary = .{
                .source_function = source_id,
                .op_ids = &wide_ops,
                .provider_names = &provider_names,
            },
            .callable = source,
            .implementations = &implementations[0],
            .measurements = &measurements[0],
        },
        .{
            .boundary = .{
                .source_function = source_id,
                .op_ids = &left_ops,
                .provider_names = &provider_names,
            },
            .callable = source,
            .implementations = &implementations[1],
            .measurements = &measurements[1],
        },
        .{
            .boundary = .{
                .source_function = source_id,
                .op_ids = &right_ops,
                .provider_names = &provider_names,
            },
            .callable = source,
            .implementations = &implementations[2],
            .measurements = &measurements[2],
        },
    };

    var resolver = LocalResolver{};
    var resolution = try resolver.interface().resolve(&program, &evaluated, testing.allocator);
    defer resolution.deinit();
    switch (resolution.decisions[0].implementation) {
        .unreplaced => {},
        .provider => return error.TestUnexpectedResult,
    }
    switch (resolution.decisions[1].implementation) {
        .provider => {},
        .unreplaced => return error.TestUnexpectedResult,
    }
    switch (resolution.decisions[2].implementation) {
        .provider => {},
        .unreplaced => return error.TestUnexpectedResult,
    }
}
