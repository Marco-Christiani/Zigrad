//! Kernel-provider selection, artifacts, and dispatch registration.
//!
//! Tuning asks providers to compile callable PR functions, optionally measures
//!  their implementations, and resolves final selections into a `KernelStore`.
//!
//! Kernelization reads the store, and execution resolves providers through a
//!  separate `DispatchRegistry`.
const std = @import("std");
const artifact_mod = @import("kernel/artifact.zig");
const device = @import("device");
const Range = @import("pr").analysis.pattern.Range;
const fingerprint = @import("pr").analysis.fingerprint;
const pr = @import("pr");
const store_mod = @import("kernel/store.zig");
const TypedPtr = @import("rtti").TypedPtr;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.@"zg/kernel");

/// Custom-call target used by kernel-provider dispatch.
pub const dispatch_target_name = "zigrad.kernel.dispatch";

/// Region annotation requesting a candidate from a named kernel provider.
pub const provider_annotation_name = "zigrad.kernel.provider";

/// Contiguous PR operation range recognized by a provider.
/// See `Range`
pub const Match = Range;

/// Invalid payloads for the provider annotation contract.
pub const AnnotationError = error{
    /// The annotation value contains no valid provider request.
    InvalidProviderAnnotation,
};

/// Invalid kernel-provider configuration.
pub const ProviderConfigError = error{
    /// A provider has an empty name.
    InvalidKernelProviderName,
    /// Two providers use the same name.
    DuplicateKernelProviderName,
};

/// Validate provider names used as selection and dispatch identities.
pub fn validate_providers(
    /// Providers that participate in one discovery or tuning operation.
    providers: []const KernelProvider,
) ProviderConfigError!void {
    for (providers, 0..) |*provider, index| {
        if (provider.name.len == 0) return error.InvalidKernelProviderName;
        for (providers[0..index]) |*prior| {
            if (std.mem.eql(u8, prior.name, provider.name)) {
                return error.DuplicateKernelProviderName;
            }
        }
    }
}

/// Construct a provider request for a region builder.
pub fn provider_annotation(
    /// Stable provider name borrowed by the returned annotation.
    provider_name: []const u8,
) pr.Annotation {
    return .{
        .name = provider_annotation_name,
        .value = .{ .string = provider_name },
    };
}

/// Construct a provider request containing every eligible provider.
pub fn providers_annotation(
    /// Stable provider names borrowed by the returned annotation.
    provider_names: []const []const u8,
) pr.Annotation {
    return .{
        .name = provider_annotation_name,
        .value = .{ .strings = provider_names },
    };
}

/// Providers eligible to implement one requested region.
pub const ProviderRequest = union(enum) {
    /// One eligible provider.
    one: []const u8,
    /// Multiple eligible providers.
    many: []const []const u8,

    /// Return the number of requested providers.
    pub fn len(self: ProviderRequest) usize {
        return switch (self) {
            .one => 1,
            .many => |names| names.len,
        };
    }

    /// Return one provider name by position.
    pub fn at(
        self: ProviderRequest,
        /// Position in the request. Values outside `len()` are invalid.
        index: usize,
    ) []const u8 {
        return switch (self) {
            .one => |name| if (index == 0) name else unreachable,
            .many => |names| names[index],
        };
    }
};

/// Callable boundary associated with providers eligible to implement it.
pub const CandidateRegion = struct {
    /// Function containing `op_ids` when discovery ran.
    source_function: pr.FunctionId,
    /// Stable IDs for the contained contiguous source ops.
    op_ids: []const u32,
    /// Provider names eligible at this boundary.
    provider_names: []const []const u8,
    /// Existing region id for an explicit candidate.
    explicit_region: ?u32 = null,

    /// Return the provider request represented by this candidate.
    pub fn request(self: CandidateRegion) ProviderRequest {
        std.debug.assert(self.provider_names.len > 0);
        return if (self.provider_names.len == 1)
            .{ .one = self.provider_names[0] }
        else
            .{ .many = self.provider_names };
    }

    /// Return whether two records identify the same source boundary.
    pub fn same_boundary(
        self: CandidateRegion,
        /// Candidate occurrence to compare by source function and op ids.
        other: CandidateRegion,
    ) bool {
        return self.source_function == other.source_function and
            std.mem.eql(u32, self.op_ids, other.op_ids);
    }

    /// Resolve retained op IDs against `func` and return their contiguous range.
    pub fn resolve_range(
        self: CandidateRegion,
        /// Function expected to contain the retained op identities.
        func: pr.Function,
    ) ?Match {
        if (self.op_ids.len == 0) return null;
        const start = func.op_index_by_id(self.op_ids[0]) orelse return null;
        const end = start + self.op_ids.len;
        if (end > func.ops.len) return null;
        for (self.op_ids, func.ops[start..end]) |op_id, op| {
            if (op_id != op.id) return null;
        }
        return .{ .start = start, .end = end };
    }
};

/// Candidate boundary paired with its extracted callable.
pub const ExtractedCandidate = struct {
    /// Discovered source boundary and eligible providers.
    boundary: CandidateRegion,
    /// Callable containing the extracted source operations.
    callable_function: pr.FunctionId,
};

/// Ephemeral provider candidates discovered for one PR program.
///
/// Entries retain overlapping boundaries so conflict resolution can occur
///  after compilation and measurement.
pub const Candidates = struct {
    /// Owns the candidate arena and supports transactional replacement.
    allocator: Allocator,
    /// Owns op-id and provider-list storage.
    arena: std.heap.ArenaAllocator,
    /// Discovered boundaries in insertion order.
    entries: std.ArrayList(CandidateRegion) = .empty,

    /// Initialize an empty candidate collection.
    pub fn init(
        /// Allocator retained by the candidate arena.
        allocator: Allocator,
    ) Candidates {
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
        };
    }

    /// Release candidate records and provider-name lists.
    pub fn deinit(self: *Candidates) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Add one provider to a candidate boundary.
    ///
    /// Equal boundaries are merged. Provider names are borrowed and must
    ///  outlive this collection.
    pub fn add(
        self: *Candidates,
        /// Function containing the candidate boundary.
        source_function: pr.FunctionId,
        /// Function value used to resolve `range` into op identities.
        func: pr.Function,
        /// Nonempty contiguous range claimed by the provider.
        range: Match,
        /// Stable provider name borrowed by this collection.
        provider_name: []const u8,
        /// Region that explicitly requested this candidate, when present.
        explicit_region: ?u32,
    ) Allocator.Error!void {
        std.debug.assert(range.start < range.end);
        std.debug.assert(range.end <= func.ops.len);
        std.debug.assert(provider_name.len > 0);
        const allocator = self.arena.allocator();
        const matched_ops = func.ops[range.start..range.end];
        for (self.entries.items) |*entry| {
            if (entry.source_function != source_function or
                !same_operations(entry.op_ids, matched_ops)) continue;
            for (entry.provider_names) |existing| {
                if (std.mem.eql(u8, existing, provider_name)) return;
            }
            const names = try allocator.alloc([]const u8, entry.provider_names.len + 1);
            @memcpy(names[0..entry.provider_names.len], entry.provider_names);
            names[entry.provider_names.len] = provider_name;
            std.mem.sort([]const u8, names, {}, string_less_than);
            entry.provider_names = names;
            if (explicit_region) |region| entry.explicit_region = region;
            return;
        }

        const op_ids = try allocator.alloc(u32, matched_ops.len);
        for (matched_ops, op_ids) |op, *op_id| op_id.* = op.id;
        const names = try allocator.alloc([]const u8, 1);
        names[0] = provider_name;
        try self.entries.append(allocator, .{
            .source_function = source_function,
            .op_ids = op_ids,
            .provider_names = names,
            .explicit_region = explicit_region,
        });
    }
};

/// Candidate boundaries whose callables have been extracted into a PR program.
///
/// The collection owns candidate records, op-id slices, and provider-name
///  lists. Provider-name bytes borrowed from program annotations or provider
///  configuration.
pub const ExtractedCandidates = struct {
    /// Allocator retained to rebuild the collection transactionally.
    allocator: Allocator,
    /// Owns candidate records and boundary slices.
    arena: std.heap.ArenaAllocator,
    /// Extracted candidates in discovery order.
    entries: std.ArrayList(ExtractedCandidate) = .empty,

    /// Initialize an empty extracted-candidate collection.
    pub fn init(
        /// Allocator owning the collection and retained for transactional rebuilds.
        allocator: Allocator,
    ) ExtractedCandidates {
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
        };
    }

    /// Release candidate records and boundary slices.
    pub fn deinit(self: *ExtractedCandidates) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Copy one boundary and associate it with its extracted callable.
    pub fn append(
        self: *ExtractedCandidates,
        /// Discovered boundary copied into this collection.
        boundary: CandidateRegion,
        /// Callable containing the extracted source operations.
        callable_function: pr.FunctionId,
    ) Allocator.Error!void {
        const allocator = self.arena.allocator();
        const op_ids = try allocator.dupe(u32, boundary.op_ids);
        const provider_names = try allocator.dupe([]const u8, boundary.provider_names);
        try self.entries.append(allocator, .{
            .boundary = .{
                .source_function = boundary.source_function,
                .op_ids = op_ids,
                .provider_names = provider_names,
                .explicit_region = boundary.explicit_region,
            },
            .callable_function = callable_function,
        });
    }
};

fn same_operations(op_ids: []const u32, ops: []const *pr.Op) bool {
    if (op_ids.len != ops.len) return false;
    for (op_ids, ops) |op_id, op| {
        if (op_id != op.id) return false;
    }
    return true;
}

fn string_less_than(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

/// Return the providers requested by an annotated IR object.
pub fn requested_providers(
    /// Annotations on the function or region being inspected.
    annotations: []const pr.Annotation,
) AnnotationError!?ProviderRequest {
    const found = for (annotations) |*annotation| {
        if (std.mem.eql(u8, annotation.name, provider_annotation_name))
            break annotation;
    } else return null;
    const request: ProviderRequest = switch (found.value) {
        .string => |name| .{ .one = name },
        .strings => |names| if (names.len == 0)
            return error.InvalidProviderAnnotation
        else
            .{ .many = names },
        else => return error.InvalidProviderAnnotation,
    };
    for (0..request.len()) |index| {
        const name = request.at(index);
        if (name.len == 0) return error.InvalidProviderAnnotation;
        for (0..index) |prior| {
            if (std.mem.eql(u8, request.at(prior), name)) return error.InvalidProviderAnnotation;
        }
    }
    return request;
}

pub const Artifact = artifact_mod.Artifact;
pub const Implementation = store_mod.Implementation;
pub const SelectionKey = store_mod.SelectionKey;
pub const KernelStore = store_mod.KernelStore;
pub const ProviderCandidate = store_mod.ProviderCandidate;
pub const PutError = store_mod.PutError;
pub const Selection = store_mod.Selection;
pub const MeasurementEvidence = store_mod.MeasurementEvidence;

/// Failures produced while encoding selection identity.
pub const IdentityError = Allocator.Error || std.Io.Writer.Error;

/// Combine eligible providers, selected device, and callable fingerprint.
///
/// This identifies a callable implementation independently of where it occurs
///  in a source program.
/// Platform names are normalized since `Platform.eql` ignores ASCII case.
/// Caller owns the result.
pub fn make_implementation_key(
    /// Allocator owning the returned key bytes.
    allocator: Allocator,
    /// Eligible provider names encoded into the key.
    providers: ProviderRequest,
    /// Target device encoded into the key.
    selected_device: device.Device,
    /// Callable fingerprint encoded into the key.
    function_fingerprint: fingerprint.Function,
) IdentityError!SelectionKey {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    const writer = &output.writer;

    try writer.print("kp3:{d}:", .{providers.len()});
    for (0..providers.len()) |index| {
        const provider_name = providers.at(index);
        try writer.print("{d}:", .{provider_name.len});
        try writer.writeAll(provider_name);
        try writer.writeByte(':');
    }
    try writer.print("{d}:", .{selected_device.platform.name.len});
    for (selected_device.platform.name) |byte| {
        try writer.writeByte(std.ascii.toLower(byte));
    }
    try writer.print(":{d}:", .{selected_device.ordinal});
    try function_fingerprint.write_hex(writer);

    return .{ .bytes = try output.toOwnedSlice() };
}

/// Identify one candidate occurrence and its eligible implementations.
///
/// Operation ids preserve the source boundary identity across transient range
///  resolution.
///
/// Caller owns the result.
pub fn make_selection_key(
    /// Allocator owning the returned key bytes.
    allocator: Allocator,
    /// Source occurrence and eligible provider names encoded into the key.
    candidate: CandidateRegion,
    /// Target device encoded into the key.
    selected_device: device.Device,
    /// Extracted callable fingerprint encoded into the key.
    function_fingerprint: fingerprint.Function,
) IdentityError!SelectionKey {
    const implementation = try make_implementation_key(
        allocator,
        candidate.request(),
        selected_device,
        function_fingerprint,
    );
    defer allocator.free(implementation.bytes);

    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    const writer = &output.writer;
    try writer.print("kp-occurrence:fn={d}:ops={d}:", .{
        @intFromEnum(candidate.source_function),
        candidate.op_ids.len,
    });
    for (candidate.op_ids) |op_id| try writer.print("{d}:", .{op_id});
    try writer.writeAll(implementation.bytes);
    return .{ .bytes = try output.toOwnedSlice() };
}

/// Append one abstract value to a diagnostic string.
pub fn write_aval_signature(
    /// Destination writer.
    writer: *std.Io.Writer,
    /// Abstract value to encode.
    aval: pr.Aval,
) !void {
    switch (aval) {
        .tensor => |tensor| {
            try writer.writeAll(@tagName(tensor.dtype));
            try writer.writeByte('[');
            for (tensor.shape.dims, 0..) |dim, index| {
                if (index > 0) try writer.writeByte(',');
                try writer.print("{d}", .{dim});
            }
            try writer.writeByte(']');
        },
    }
}

pub const DType = @import("dtype").DType;

/// Descriptor for a single buffer passed through an FFI call.
///
/// The execution integration extracts these descriptors from FFI frames.
pub const BufferDesc = struct {
    data: *anyopaque,
    dtype: DType,
    dims: []const i64,
    rank: usize,
};

/// Context passed to a provider's dispatch function at execution time.
///
/// It supplies buffers, device identity, an optional device stream, and workspace.
///
/// TODO(kernel-provider): Pass a device allocator when integrations expose one.
pub const DispatchContext = struct {
    /// Input buffers in callable parameter order.
    inputs: []const BufferDesc,
    /// Output buffers in callable return order.
    outputs: []const BufferDesc,
    /// Device executing this invocation.
    device: device.Device,

    /// Platform stream handle received from the integration FFI, or null on host.
    stream: ?*anyopaque = null,

    /// Provider workspace, or null when the artifact requires none.
    workspace: ?*anyopaque = null,

    /// Compile-time workspace requirement reported by the provider artifact.
    workspace_bytes_required: usize,

    /// Allocator available for host-side dispatch work.
    allocator: std.mem.Allocator,
};

/// Failures exposed by kernel-provider dispatch.
pub const DispatchError = error{
    /// Dispatch failed, provider logged details.
    DispatchFailed,
    /// Artifact or buffers use a data type unsupported by the provider.
    UnsupportedDType,
    /// Artifact and invocation buffer shapes do not agree.
    ShapeMismatch,
    /// Provider cannot execute on the selected device.
    UnsupportedDevice,
    /// Provider runtime could not be loaded.
    ProviderLoadFailed,
    /// Required provider workspace was not supplied.
    WorkspaceUnavailable,
} || Allocator.Error;

/// Context available while preparing provider artifacts for execution.
pub const PrepareContext = struct {
    /// Device selected for the upcoming execution.
    device: device.Device,
};

/// Failures exposed while preparing kernel artifacts for execution.
pub const PrepareError = error{
    /// The store references a provider absent from the dispatch registry.
    ProviderNotRegistered,
    /// Artifact preparation failed, provider logged details.
    PrepareFailed,
    /// Provider runtime could not be loaded.
    ProviderLoadFailed,
    /// Provider cannot prepare the artifact for the selected device.
    UnsupportedDevice,
} || Allocator.Error;

/// Provider artifact-preparation function signature.
///
/// Preparation creates process-local runtime state from portable artifact
///  bytes. Callers invoke it before execution begins.
pub const PrepareFn = *const fn (
    provider_ctx: TypedPtr,
    artifact_data: []const u8,
    kernel_key: []const u8,
    ctx: PrepareContext,
) PrepareError!void;

/// Provider dispatch function signature.
///
/// Called by an integration when a custom call targets a kernelized op.
pub const DispatchFn = *const fn (
    /// Provider state.
    provider_ctx: TypedPtr,
    /// Opaque provider artifact bytes.
    artifact_data: []const u8,
    /// Key to identify the compiled kernel.
    kernel_key: []const u8,
    ctx: DispatchContext,
) DispatchError!void;

/// Failures in kernel-provider compilation.
pub const CompileError = error{
    /// Provider cannot handle this function.
    Unsupported,
    /// Compilation failed, provider logged details.
    CompileFailed,
    /// Provider runtime could not be loaded.
    ProviderLoadFailed,
    /// Provider API call failed or returned unexpected data.
    ProviderCallFailed,
} || Allocator.Error;

/// Failures produced while a provider discovers supported operation boundaries.
pub const DiscoverError = Allocator.Error;

/// Runtime-polymorphic provider discovery and compilation.
pub const ProviderCompiler = struct {
    /// Borrowed provider state.
    context: *anyopaque,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Compile one callable PR function for the selected device.
        compile: *const fn (
            context: *anyopaque,
            func: pr.Function,
            selected_device: device.Device,
            allocator: Allocator,
        ) CompileError!Artifact,
        /// Append supported nonempty contiguous operation ranges.
        discover: ?*const fn (
            context: *anyopaque,
            func: pr.Function,
            matches: *std.ArrayList(Match),
            allocator: Allocator,
        ) DiscoverError!void = null,
    };

    /// Compile one callable PR function for the selected device.
    pub fn compile(
        self: ProviderCompiler,
        /// Extracted callable to compile.
        func: pr.Function,
        /// Target device selected for the artifact.
        selected_device: device.Device,
        /// Allocator owning returned artifact bytes.
        allocator: Allocator,
    ) CompileError!Artifact {
        return try self.vtable.compile(self.context, func, selected_device, allocator);
    }

    /// Append every operation boundary recognized by the provider.
    pub fn discover(
        self: ProviderCompiler,
        /// Function inspected for supported ranges.
        func: pr.Function,
        /// Destination for discovered ranges.
        matches: *std.ArrayList(Match),
        /// Allocator used to grow `matches`.
        allocator: Allocator,
    ) DiscoverError!void {
        const discover_fn = self.vtable.discover orelse return;
        return try discover_fn(self.context, func, matches, allocator);
    }
};

/// Process-local runtime capability for provider artifacts.
pub const ProviderRuntime = struct {
    /// Type-tagged provider state borrowed by the runtime.
    context: TypedPtr,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Dispatch one prepared provider artifact.
        dispatch: DispatchFn,
        /// Prepare portable artifact bytes for process-local execution.
        prepare: ?PrepareFn = null,
    };

    /// Prepare an artifact when this runtime requires process-local state.
    pub fn prepare(
        self: ProviderRuntime,
        /// Portable provider artifact bytes.
        artifact_data: []const u8,
        /// Selection key identifying the artifact in this process.
        kernel_key: []const u8,
        /// Device selected for the upcoming execution.
        ctx: PrepareContext,
    ) PrepareError!void {
        const prepare_fn = self.vtable.prepare orelse return;
        return try prepare_fn(self.context, artifact_data, kernel_key, ctx);
    }

    /// Dispatch one provider artifact through this runtime.
    pub fn dispatch(
        self: ProviderRuntime,
        /// Portable provider artifact bytes.
        artifact_data: []const u8,
        /// Selection key identifying the artifact in this process.
        kernel_key: []const u8,
        /// Buffers, device, stream, and workspace for this invocation.
        ctx: DispatchContext,
    ) DispatchError!void {
        return try self.vtable.dispatch(self.context, artifact_data, kernel_key, ctx);
    }
};

/// Named provider capabilities used by tuning and execution.
pub const KernelProvider = struct {
    /// Stable provider name used in selection and dispatch keys.
    name: []const u8,
    /// Discovery and compilation capability.
    compiler: ProviderCompiler,
    /// Execution capability. Null when this process only compiles artifacts.
    runtime: ?ProviderRuntime = null,
};

/// Find a configured provider by its stable name.
pub fn find_provider(
    /// Provider configuration to search.
    providers: []const KernelProvider,
    /// Exact provider name to resolve.
    name: []const u8,
) ?*const KernelProvider {
    for (providers) |*provider| {
        if (std.mem.eql(u8, provider.name, name)) return provider;
    }
    return null;
}

/// Maps provider names to runtime capabilities used during execution.
///
/// Applications populate the registry before artifact preparation. The kernel
///  store contains no function pointers.
pub const DispatchRegistry = struct {
    /// Allocator owning the registry table.
    allocator: std.mem.Allocator,
    /// Runtime capabilities keyed by borrowed provider names.
    entries: std.StringHashMapUnmanaged(ProviderRuntime) = .{},

    /// Initialize an empty dispatch registry.
    pub fn init(reg_allocator: std.mem.Allocator) DispatchRegistry {
        return .{ .allocator = reg_allocator };
    }

    /// Release the registry table.
    pub fn deinit(self: *DispatchRegistry) void {
        self.entries.deinit(self.allocator);
    }

    /// Register runtime hooks exposed by `providers`.
    pub fn register_providers(
        self: *DispatchRegistry,
        /// Providers whose available runtime capabilities are registered.
        providers: []const KernelProvider,
    ) (ProviderConfigError || Allocator.Error)!void {
        try validate_providers(providers);
        for (providers) |*provider| {
            const runtime = provider.runtime orelse continue;
            try self.register(provider.name, runtime);
        }
    }

    /// Register or replace a provider's runtime capability.
    pub fn register(
        self: *DispatchRegistry,
        /// Borrowed name that must outlive the registry entry.
        provider_name: []const u8,
        /// Runtime capability whose context must outlive the registry entry.
        entry: ProviderRuntime,
    ) Allocator.Error!void {
        if (self.entries.contains(provider_name)) {
            log.info("dispatch registry replacing provider '{s}' entry", .{provider_name});
        }
        try self.entries.put(self.allocator, provider_name, entry);
    }

    /// Look up a runtime capability by provider name.
    pub fn get(
        self: *const DispatchRegistry,
        /// Provider name to resolve.
        provider_name: []const u8,
    ) ?ProviderRuntime {
        return self.entries.get(provider_name);
    }

    /// Prepare every selected provider artifact in `store` for execution.
    ///
    /// Providers without a preparation hook consume their portable artifact
    ///  bytes directly during dispatch. Preparation order is unspecified.
    pub fn prepare(
        self: *const DispatchRegistry,
        /// Final selections whose provider artifacts are prepared.
        store: *const KernelStore,
        /// Device shared by every selected artifact.
        ctx: PrepareContext,
    ) PrepareError!void {
        var selections = store.selections.iterator();
        while (selections.next()) |selection| {
            const stored = switch (selection.value_ptr.candidate) {
                .provider => |value| value,
                .unreplaced => continue,
            };
            const entry = self.get(stored.provider_name) orelse
                return error.ProviderNotRegistered;
            try entry.prepare(stored.artifact.data, selection.key_ptr.*, ctx);
        }
    }
};

test validate_providers {
    const provider = provider_for_test("duplicate");
    try std.testing.expectError(
        error.DuplicateKernelProviderName,
        validate_providers(&.{ provider, provider }),
    );
}

test make_implementation_key {
    const testing = std.testing;
    const function_fingerprint = fingerprint.Function{ .bytes = .{0x5a} ** 32 };
    const providers = ProviderRequest{ .many = &.{ "tvm", "mirage" } };

    const first = try make_implementation_key(
        testing.allocator,
        providers,
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(first.bytes);
    const same_candidate_set = try make_implementation_key(
        testing.allocator,
        providers,
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(same_candidate_set.bytes);
    const other_device = try make_implementation_key(
        testing.allocator,
        providers,
        .{ .platform = .cuda, .ordinal = 1 },
        function_fingerprint,
    );
    defer testing.allocator.free(other_device.bytes);
    const reported_case = try make_implementation_key(
        testing.allocator,
        providers,
        .{ .platform = .{ .name = "CUDA" } },
        function_fingerprint,
    );
    defer testing.allocator.free(reported_case.bytes);

    try testing.expectEqualStrings(first.bytes, same_candidate_set.bytes);
    try testing.expect(!std.mem.eql(u8, first.bytes, other_device.bytes));
    try testing.expectEqualStrings(first.bytes, reported_case.bytes);
}

test make_selection_key {
    const testing = std.testing;
    const function_fingerprint = fingerprint.Function{ .bytes = .{0x5a} ** 32 };
    const provider_names = [_][]const u8{"tvm"};
    const first_candidate = CandidateRegion{
        .source_function = @enumFromInt(7),
        .op_ids = &.{ 11, 12 },
        .provider_names = &provider_names,
    };
    const second_candidate = CandidateRegion{
        .source_function = @enumFromInt(7),
        .op_ids = &.{ 21, 22 },
        .provider_names = &provider_names,
    };

    const first = try make_selection_key(
        testing.allocator,
        first_candidate,
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(first.bytes);
    const same = try make_selection_key(
        testing.allocator,
        first_candidate,
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(same.bytes);
    const second = try make_selection_key(
        testing.allocator,
        second_candidate,
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(second.bytes);

    try testing.expectEqualStrings(first.bytes, same.bytes);
    try testing.expect(!std.mem.eql(u8, first.bytes, second.bytes));
}

test "dispatch registry register and get" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var dummy_ctx: u8 = 42;
    try registry.register("mirage", .{
        .context = TypedPtr.init(&dummy_ctx),
        .vtable = &.{ .dispatch = Dummy.dispatch },
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u8, 42), entry.context.cast(u8).*);
    try testing.expect(registry.get("tvm") == null);
}

test "dispatch registry replaces duplicate" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var ctx1: u8 = 1;
    var ctx2: u8 = 2;
    try registry.register("mirage", .{
        .context = TypedPtr.init(&ctx1),
        .vtable = &.{ .dispatch = Dummy.dispatch },
    });
    try registry.register("mirage", .{
        .context = TypedPtr.init(&ctx2),
        .vtable = &.{ .dispatch = Dummy.dispatch },
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u8, 2), entry.context.cast(u8).*);
}

test "dispatch registry prepares selected provider artifacts" {
    const testing = std.testing;

    const State = struct {
        calls: usize = 0,

        fn prepare(
            provider_ctx: TypedPtr,
            artifact_data: []const u8,
            kernel_key: []const u8,
            ctx: PrepareContext,
        ) PrepareError!void {
            const self = provider_ctx.cast(@This());
            if (!std.mem.eql(u8, "compiled_kernel_bytes", artifact_data))
                return error.PrepareFailed;
            if (!std.mem.eql(u8, "kp-test:prepare", kernel_key))
                return error.PrepareFailed;
            if (!ctx.device.platform.eql(.cuda)) return error.UnsupportedDevice;
            self.calls += 1;
        }

        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try put_provider_for_test(&store, "kp-test:prepare", "test", "compiled_kernel_bytes");
    try store.put(.{ .bytes = "kp-test:unreplaced" }, .{
        .candidate = .unreplaced,
        .reason = "unsupported",
    });

    var state: State = .{};
    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();
    try registry.register("test", .{
        .context = TypedPtr.init(&state),
        .vtable = &.{
            .dispatch = State.dispatch,
            .prepare = State.prepare,
        },
    });

    try registry.prepare(&store, .{ .device = .{ .platform = .cuda } });
    try testing.expectEqual(@as(usize, 1), state.calls);
}

test "dispatch registry requires providers for stored artifacts" {
    const testing = std.testing;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try put_provider_for_test(
        &store,
        "kp-test:missing-provider",
        "missing",
        "compiled_kernel_bytes",
    );

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();
    try testing.expectError(
        error.ProviderNotRegistered,
        registry.prepare(&store, .{ .device = .{ .platform = .cuda } }),
    );
}

fn put_provider_for_test(
    store: *KernelStore,
    key: []const u8,
    provider_name: []const u8,
    data: []const u8,
) !void {
    try store.put(.{ .bytes = key }, .{
        .candidate = .{ .provider = .{
            .provider_name = provider_name,
            .artifact = .{ .data = try store.allocator().dupe(u8, data) },
        } },
        .reason = "available",
    });
}

fn provider_for_test(name: []const u8) KernelProvider {
    const Compiler = struct {
        fn compile(
            _: *anyopaque,
            _: pr.Function,
            _: device.Device,
            _: Allocator,
        ) CompileError!Artifact {
            return error.Unsupported;
        }
    };
    return .{
        .name = name,
        .compiler = .{
            .context = @constCast(&{}),
            .vtable = &.{ .compile = Compiler.compile },
        },
    };
}
