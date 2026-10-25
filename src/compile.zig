//! Typed Tensor invocation over tracing and the assembled ZC CPU composition.

const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("tensor.zig");
const trace = @import("trace.zig");
const meta = @import("utils.zig").meta;
const pr = @import("pr");
const zc = @import("zc");
const zcf = @import("zcf");
const implementation_manifest = @import("zc_implementation_manifest");

const Id = zc.schema.Id;
const standard = zc.x86_64.installation;
const Buffer = zc.x86_64.execute.Buffer;
const log = std.log.scoped(.@"zg/compile");

fn validation_error(err: anyerror, comptime context: []const u8) anyerror {
    if (comptime builtin.is_test) {
        if (TestDiagnostics.expected) |*expected| {
            if (err == expected.cause and std.mem.eql(u8, context, expected.context)) {
                expected.count += 1;
                return err;
            }
        }
    }
    log.err("{s}: {s}", .{ context, @errorName(err) });
    return err;
}

const TestDiagnostics = if (builtin.is_test) struct {
    const Expected = struct {
        cause: anyerror,
        context: []const u8,
        count: usize = 0,
    };
    threadlocal var expected: ?Expected = null;
} else struct {};

fn expect_validation_error(
    comptime cause: anyerror,
    comptime context: []const u8,
    comptime operation: anytype,
    args: anytype,
) !void {
    std.debug.assert(TestDiagnostics.expected == null);
    TestDiagnostics.expected = .{ .cause = cause, .context = context };
    defer TestDiagnostics.expected = null;
    try std.testing.expectError(cause, @call(.auto, operation, args));
    try std.testing.expectEqual(@as(usize, 1), TestDiagnostics.expected.?.count);
}

/// Checks the execution target before tracing or creating a workspace.
fn host_admitted(os: std.Target.Os.Tag, arch: std.Target.Cpu.Arch) bool {
    return os == .linux and arch == .x86_64;
}

/// Accepts concrete Tensor parameters in a flat tuple and one Tensor result.
fn signature_supported(comptime func: anytype, comptime Specs: type) bool {
    const info = @typeInfo(@TypeOf(func));
    if (info != .@"fn") return false;
    const f = info.@"fn";
    if (f.is_generic or f.attrs.varargs) return false;
    const tuple = @typeInfo(Specs);
    if (tuple != .@"struct" or !tuple.@"struct".is_tuple) return false;
    const fields = tuple.@"struct".field_types;
    if (fields.len != f.param_types.len) return false;
    inline for (fields, f.param_types) |field, param| {
        if (field != Tensor or param != Tensor) return false;
    }
    const result = f.return_type orelse return false;
    return result == Tensor or (@typeInfo(result) == .error_union and
        @typeInfo(result).error_union.payload == Tensor);
}

/// Copies logical dtype, shape, and byte capacity from source values.
const Descriptor = struct {
    dtype: pr.DType,
    shape: pr.BoundedShape,
    byte_count: usize,

    fn init(dtype: pr.DType, dims: []const i64) !Descriptor {
        if (dims.len > Tensor.max_rank) return validation_error(error.UnsupportedRank, "source rank exceeds Tensor capacity");
        var count: usize = 1;
        for (dims) |dim| {
            const size = std.math.cast(usize, dim) orelse return validation_error(error.InvalidShape, "source dimension must fit usize");
            count = try std.math.mul(usize, count, size);
        }
        return .{
            .dtype = dtype,
            .shape = .from_slice(dims),
            .byte_count = try std.math.mul(usize, count, dtype.size_in_bytes()),
        };
    }

    fn from_value(value: *const pr.Var) !Descriptor {
        const tensor = value.aval.as_tensor();
        return init(tensor.dtype, tensor.shape.dims);
    }

    /// Checks both metadata layers before obtaining a byte view.
    fn input_bytes(self: Descriptor, tensor: Tensor) ![]const u8 {
        if (tensor.dtype != self.dtype) return validation_error(error.DTypeMismatch, "call input dtype disagrees with compiled descriptor");
        if (tensor.shape.len > Tensor.max_rank) return validation_error(error.ShapeMismatch, "call input shape disagrees with compiled descriptor");
        if (!std.mem.eql(i64, tensor.shape.const_slice(), self.shape.const_slice()))
            return validation_error(error.ShapeMismatch, "call input shape disagrees with compiled descriptor");
        const host = switch (tensor.backing) {
            .host => |host| host,
            .abstract, .device, .traced => return validation_error(error.UnsupportedBacking, "call input requires host backing"),
        };
        if (host.shape.len > Tensor.max_rank or host.dtype != tensor.dtype or
            !std.mem.eql(i64, host.shape.const_slice(), tensor.shape.const_slice()))
            return validation_error(error.HostMetadataMismatch, "call input host metadata disagrees with Tensor metadata");
        const bytes = host.data();
        if (bytes.len < self.byte_count) return validation_error(error.BufferTooSmall, "call input capacity is below the compiled byte count");
        return bytes[0..self.byte_count];
    }
};

/// Writable storage aligned to one external image binding.
const Storage = struct {
    raw: []u8,
    bytes: []u8,

    fn init(allocator: std.mem.Allocator, size: usize, alignment: usize) !Storage {
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return validation_error(error.InvalidAlignment, "external binding alignment must be a representable power of two");
        const raw = try allocator.alloc(u8, try std.math.add(usize, size, alignment - 1));
        const offset = std.mem.alignForward(usize, @intFromPtr(raw.ptr), alignment) - @intFromPtr(raw.ptr);
        return .{ .raw = raw, .bytes = raw[offset..][0..size] };
    }

    fn deinit(self: Storage, allocator: std.mem.Allocator) void {
        allocator.free(self.raw);
    }
};

/// Owns the writable copies passed through the native byte boundary.
const Bindings = struct {
    storage: []?Storage,
    inputs: []Buffer,
    output: Buffer,

    fn init(allocator: std.mem.Allocator, image: *const zc.x86_64.artifact.Image, inputs: []const Descriptor, output: Descriptor) !Bindings {
        if (image.input_count != inputs.len or image.output_count != 1) return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
        const node_count = inputs.len + 1;
        const parents = try allocator.alloc(usize, node_count);
        defer allocator.free(parents);
        for (parents, 0..) |*parent, node| parent.* = node;
        for (image.buffers) |buffer| switch (buffer.storage) {
            .allocation => {},
            .external => |external| {
                var anchor: ?usize = null;
                for (external.inputs) |position| {
                    if (position >= inputs.len) return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
                    merge(parents, &anchor, position);
                }
                for (external.outputs) |position| {
                    if (position != 0) return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
                    merge(parents, &anchor, inputs.len);
                }
            },
        };
        const storage = try allocator.alloc(?Storage, node_count);
        @memset(storage, null);
        errdefer {
            for (storage) |item| if (item) |owned| owned.deinit(allocator);
            allocator.free(storage);
        }
        const views = try allocator.alloc(Buffer, inputs.len);
        errdefer allocator.free(views);
        // each equivalence class owns one allocation and exposes identical views
        for (0..node_count) |root| {
            if (find(parents, root) != root) continue;
            var capacity: usize = 0;
            var alignment: usize = 1;
            for (0..node_count) |node| {
                if (find(parents, node) != root) continue;
                const descriptor = if (node < inputs.len) inputs[node] else output;
                const required_alignment = if (node < inputs.len)
                    try external_alignment(image, .input, node, descriptor.byte_count)
                else
                    try external_alignment(image, .output, 0, descriptor.byte_count);
                capacity = @max(capacity, descriptor.byte_count);
                alignment = @max(alignment, required_alignment);
            }
            storage[root] = try Storage.init(allocator, capacity, alignment);
        }
        for (views, 0..) |*view, node|
            view.* = .{ .bytes = storage[find(parents, node)].?.bytes };
        return .{
            .storage = storage,
            .inputs = views,
            .output = .{ .bytes = storage[find(parents, inputs.len)].?.bytes },
        };
    }

    fn find(parents: []const usize, node: usize) usize {
        var root = node;
        while (parents[root] != root) root = parents[root];
        return root;
    }

    fn merge(parents: []usize, anchor: *?usize, node: usize) void {
        if (anchor.*) |first_node| {
            parents[find(parents, node)] = find(parents, first_node);
        } else anchor.* = node;
    }

    fn deinit(self: Bindings, allocator: std.mem.Allocator) void {
        for (self.storage) |item| if (item) |owned| owned.deinit(allocator);
        allocator.free(self.inputs);
        allocator.free(self.storage);
    }
};

fn external_alignment(image: *const zc.x86_64.artifact.Image, direction: enum { input, output }, position: usize, size: usize) !usize {
    var alignment: usize = 1;
    var found = false;
    for (image.buffers, image.buffer_bytes) |buffer, bytes| switch (buffer.storage) {
        .allocation => {},
        .external => |external| {
            const positions = if (direction == .input) external.inputs else external.outputs;
            for (positions) |index| if (index == position) {
                if (bytes > size) return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
                alignment = @max(alignment, std.math.cast(usize, external.alignment) orelse return validation_error(error.InvalidAlignment, "external binding alignment must be a representable power of two"));
                found = true;
            };
        },
    };
    if (!found) return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
    return alignment;
}

/// Owns the source, accepted records, resolution, and native resources.
///
/// The allocator and I/O service are borrowed until `deinit`. Call inputs are
///  borrowed for each call. Calls, repeated demands, and teardown require external
///  serialization. Each result owns separate host storage and may outlive the
///  callable while its allocator remains alive.
pub const Callable = struct {
    allocator: std.mem.Allocator,
    source: pr.Program,
    function_id: pr.FunctionId,
    workspace: zc.Workspace,
    compiler: zc.Compiler,
    composition: standard.Composition,
    loaded: zc.program.LoadedProgram,
    function: zc.program.LoadedFunction,
    inputs: []Descriptor,
    output: Descriptor,
    demand: standard.Request,
    run: zcf.pull.RunResult,
    state: union(enum) {
        unprepared,
        native: struct {
            selected: Id,
            offered_contract: Id,
            opened: zc.x86_64.executable.Opened,
            prepared: zc.x86_64.execute.Prepared,
            bindings: Bindings,
        },
    } = .unprepared,

    /// Borrows inspection handles from this callable until `deinit`.
    pub fn inspect(self: *const Callable) struct {
        /// Authored PR program owned by the callable.
        source: *const pr.Program,
        /// Traced entry function in `source`.
        function_id: pr.FunctionId,
        /// Accepted compilation records owned by the workspace.
        db: *const zc.schema.Db,
        /// Accepted source subject used by the terminal demand.
        source_subject: Id,
        /// Contract required by the terminal demand.
        required_contract: Id,
        /// Retained terminal Request identity.
        request: Id,
        /// Actual Run with its resolution, termination, and callback failure evidence.
        run: *const zcf.pull.RunResult,
        /// Prepared resources and Product identities, or null when unprepared.
        native: ?struct {
            /// Sole Product selected by the terminal demand.
            selected_product: Id,
            /// Executable Contract offered by the selected Product.
            offered_contract: Id,
            /// Image borrowed by Prepared at its stable address.
            image: *const zc.x86_64.artifact.Image,
            /// Address of the executable mapping retained across calls.
            mapping: usize,
            /// Address of the native entry retained across calls.
            entry: usize,
        },
    } {
        return .{
            .source = &self.source,
            .function_id = self.function_id,
            .db = &self.workspace.db,
            .source_subject = self.function.subject,
            .required_contract = self.demand.contract.contract.id,
            .request = self.demand.request,
            .run = &self.run,
            .native = switch (self.state) {
                .unprepared => null,
                .native => |native| .{
                    .selected_product = native.selected,
                    .offered_contract = native.offered_contract,
                    .image = native.prepared.image,
                    .mapping = @intFromPtr(native.prepared.mapping.ptr),
                    .entry = @intFromPtr(native.prepared.entry),
                },
            },
        };
    }

    pub fn deinit(self: *Callable) void {
        const allocator = self.allocator;
        switch (self.state) {
            .unprepared => {},
            .native => |*native| {
                native.bindings.deinit(allocator);
                native.prepared.deinit();
                native.opened.deinit();
            },
        }
        self.run.deinit(allocator);
        allocator.free(self.inputs);
        self.loaded.deinit();
        self.compiler.deinit();
        self.workspace.deinit();
        self.source.deinit();
        allocator.destroy(self);
    }

    /// Validates all inputs before copying bytes or entering retained code.
    pub fn call(self: *Callable, args: anytype) !Tensor {
        if (comptime !tensor_tuple(@TypeOf(args))) return validation_error(error.UnsupportedSignature, "call requires a flat Tensor tuple");
        var flat: [meta.leaf_count(Tensor, @TypeOf(args))]Tensor = undefined;
        var index: usize = 0;
        meta.flatten(Tensor, @TypeOf(args), args, &flat, &index);
        if (flat.len != self.inputs.len) return validation_error(error.ArgumentCountMismatch, "call argument count disagrees with compiled inputs");
        const native = switch (self.state) {
            .native => |*native| native,
            .unprepared => {
                if (self.run.failure) |failure| return validation_error(failure.cause(), "retained CPU Run failed");
                const root = try root_resolution(&self.run, self.demand.request);
                return switch (root.resolution) {
                    .unsupported => validation_error(error.UnsupportedExecutable, "terminal CPU resolution is unsupported"),
                    .open => validation_error(error.UnresolvedExecutable, "terminal CPU resolution is open"),
                    .conflicted => validation_error(error.ConflictedContract, "terminal CPU resolution is conflicted"),
                    .satisfied => validation_error(error.UnfinishedExecutable, "terminal CPU resolution is satisfied"),
                };
            },
        };
        const bindings = &native.bindings;
        // Validate the whole tuple before the first byte copy
        for (flat, 0..) |arg, i| _ = try self.inputs[i].input_bytes(arg);
        for (flat, 0..) |arg, i|
            @memcpy(bindings.inputs[i].bytes[0..self.inputs[i].byte_count], try self.inputs[i].input_bytes(arg));
        var result = try Tensor.host(self.output.dtype, self.output.shape.const_slice(), .{ .alloc = self.allocator });
        errdefer result.deinit();
        native.prepared.run(bindings.inputs, &.{.{ .bytes = bindings.output.bytes }}) catch |err|
            return validation_error(err, "native CPU invocation rejected");
        @memcpy(result.host_data_mut(), bindings.output.bytes[0..self.output.byte_count]);
        return result;
    }

    /// Repeats the terminal demand over this accepted source and Catalog.
    ///  The caller owns the returned RunResult.
    pub fn repeat_demand(self: *Callable) !struct { demand: standard.Request, run: zcf.pull.RunResult } {
        const demand = try self.composition.request(&self.compiler, self.function.subject, .{});
        return .{ .demand = demand, .run = try self.compiler.run(&.{demand.request}, .{
            .policy = try self.compiler.accept(.{ .pull_run_policy = .catalog_closed }),
        }) };
    }
};

fn tensor_tuple(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .@"struct" or !info.@"struct".is_tuple) return false;
    inline for (info.@"struct".field_types) |field| if (field != Tensor) return false;
    return true;
}

/// Traces once, demands one CPU Product, and prepares it for repeated calls.
///
/// Initialization errors release partial ownership. Unsupported, open,
///  conflicted, unfinished, and callback-failed Runs retain an inspectable,
///  unprepared callable. The callable borrows allocator and I/O until teardown
///  and borrows inputs during each call. Calls, repeated
///  demands, and teardown require external serialization. Results own separate
///  host storage and may outlive the callable while their allocator is alive.
///
/// The default installs `default_formation` with `.reference_required` source
///  association and requests baseline x86-64 SysV callable ports with empty
///  donation permissions. It matches compatibly, retains a `catalog_closed` Run,
///  and uses `Profile.exact_scalar` with per-input executable offers.
///  Tree and tensor-plan formations are disabled.
pub fn compile(comptime func: anytype, allocator: std.mem.Allocator, io: std.Io, specs: anytype, comptime opts: trace.Options) !*Callable {
    if (comptime !signature_supported(func, @TypeOf(specs))) return validation_error(error.UnsupportedSignature, "compile requires concrete Tensor parameters and one Tensor result");
    if (!host_admitted(builtin.os.tag, builtin.cpu.arch)) return validation_error(error.UnsupportedHost, "compile requires Linux x86-64");
    const self = try allocator.create(Callable);
    errdefer allocator.destroy(self);
    self.* = .{
        .allocator = allocator,
        .source = undefined,
        .function_id = undefined,
        .workspace = undefined,
        .compiler = undefined,
        .composition = undefined,
        .loaded = undefined,
        .function = undefined,
        .inputs = undefined,
        .output = undefined,
        .demand = undefined,
        .run = undefined,
    };
    const traced = try trace.trace(func, allocator, specs, opts);
    self.source = traced.program;
    self.function_id = traced.function;
    errdefer self.source.deinit();
    self.workspace = try zc.Workspace.init(allocator, io, .memory, .{});
    errdefer self.workspace.deinit();
    self.compiler = try zc.Compiler.init(allocator, &self.workspace.db, "zigrad/compile");
    errdefer self.compiler.deinit();
    _ = try self.compiler.bind_implementation_revision(implementation_manifest.get(.fallback));
    self.composition = try standard.install(&self.compiler, .{});
    self.loaded = try zc.program.load(&self.workspace.db, allocator, &self.source, self.compiler.authority);
    errdefer self.loaded.deinit();
    self.function = try find_function(self);
    const function = self.source.get_function_by_id(self.function_id) orelse return validation_error(error.MissingSource, "traced entry function is absent from source");
    self.inputs = try allocator.alloc(Descriptor, function.params.len);
    errdefer allocator.free(self.inputs);
    for (function.params, self.inputs) |param, *descriptor| descriptor.* = try Descriptor.from_value(param);
    if (function.returns.len != 1) return validation_error(error.UnsupportedSignature, "loaded function requires one Tensor result");
    self.output = try Descriptor.from_value(function.returns[0]);
    try verify_interface(self, function);
    self.demand = try self.composition.request(&self.compiler, self.function.subject, .{});
    self.run = try self.compiler.run(&.{self.demand.request}, .{
        .policy = try self.compiler.accept(.{ .pull_run_policy = .catalog_closed }),
    });
    errdefer self.run.deinit(allocator);
    if (self.run.failure != null or self.run.termination != .resolved) return self;
    const root = try root_resolution(&self.run, self.demand.request);
    if (root.resolution != .satisfied) return self;
    const products = root.resolution.satisfied.products;
    if (products.len != 1) return validation_error(error.ExpectedOneProduct, "terminal CPU resolution requires one Product");
    const product = zc.record.typed_at(.pull_product, &self.workspace.db, products[0], .latest) orelse return validation_error(error.MissingProduct, "selected CPU Product is absent from accepted state");
    self.state = .{ .native = .{
        .selected = products[0],
        .offered_contract = product.contract,
        .opened = try self.composition.open(&self.compiler, self.demand, products[0]),
        .prepared = undefined,
        .bindings = undefined,
    } };
    // Initialize at the final address before Prepared borrows the image
    const native = &self.state.native;
    errdefer native.opened.deinit();
    native.prepared = try zc.x86_64.execute.Prepared.init(allocator, &native.opened.image.image);
    errdefer native.prepared.deinit();
    native.bindings = try Bindings.init(allocator, &native.opened.image.image, self.inputs, self.output);
    return self;
}

fn root_resolution(run: *const zcf.pull.RunResult, request: Id) !*const zcf.pull.RootResolution {
    for (run.roots) |*root| if (root.request.eql(request)) return root;
    return validation_error(error.MissingRoot, "terminal Request is absent from retained Run roots");
}

fn find_function(self: *const Callable) !zc.program.LoadedFunction {
    for (self.loaded.functions) |function| {
        const stored = zc.record.typed_at(.function, &self.workspace.db, function.record, .latest) orelse return validation_error(error.MissingFunction, "loaded source function is absent from accepted state");
        if (stored.source_id == @backingInt(self.function_id)) return function;
    }
    return validation_error(error.MissingFunction, "loaded source function is absent from accepted state");
}

fn verify_interface(self: *const Callable, function: pr.Function) !void {
    var projected = try zc.program.project(&self.workspace.db, self.allocator, self.loaded.root, self.loaded.cut, .{});
    defer projected.deinit();
    const accepted = projected.get_function_by_id(self.function_id) orelse return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
    if (accepted.params.len != function.params.len or accepted.returns.len != function.returns.len)
        return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
    for (accepted.params, function.params) |stored, source| {
        try verify_value(stored, source);
    }
    for (accepted.returns, function.returns) |stored, source| {
        try verify_value(stored, source);
    }
}

fn verify_value(stored: *const pr.Var, source: *const pr.Var) !void {
    const accepted = try Descriptor.from_value(stored);
    const authored = try Descriptor.from_value(source);
    if (accepted.dtype != authored.dtype or
        !std.mem.eql(i64, accepted.shape.const_slice(), authored.shape.const_slice()))
        return validation_error(error.InterfaceMismatch, "accepted image or source interface disagrees with compiled descriptors");
}

/// Exercises adapter buffer cleanup at every allocation boundary using a real
///  compiled image. Compiler and store allocations retain their normal allocator.
fn check_binding_allocation_failures(callable: *const Callable) !void {
    const Check = struct {
        fn bindings(allocator: std.mem.Allocator, owner: *const Callable) !void {
            const value = try Bindings.init(allocator, &owner.state.native.opened.image.image, owner.inputs, owner.output);
            defer value.deinit(allocator);
        }
        fn preparation(allocator: std.mem.Allocator, owner: *const Callable) !void {
            var value = try zc.x86_64.execute.Prepared.init(allocator, &owner.state.native.opened.image.image);
            defer value.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Check.bindings, .{callable});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Check.preparation, .{callable});
}

/// Authored source, evaluated by resident tracing with abstract Tensor specs.
fn expression(a: Tensor, b: Tensor, c: Tensor) !Tensor {
    const ab = try a.mm(b);
    const sum = try ab.add(c);
    return sum.mul(c);
}

fn test_specs(dtype: pr.DType) @TypeOf(.{ Tensor.abstract(.f32, &.{ 2, 2 }), Tensor.abstract(.f32, &.{ 2, 2 }), Tensor.abstract(.f32, &.{ 2, 2 }) }) {
    const spec = Tensor.abstract(dtype, &.{ 2, 2 });
    return .{ spec, spec, spec };
}

const Fixture = struct {
    a: [4]f32,
    b: [4]f32,
    c: [4]f32,
    expected: [4]f32,

    fn args(self: *const Fixture) !@TypeOf(test_specs(.f32)) {
        return .{
            try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&self.a) }),
            try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&self.b) }),
            try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(&self.c) }),
        };
    }
};

const first = Fixture{ .a = .{ 1, 2, 3, 4 }, .b = .{ 5, 6, 7, 8 }, .c = .{ 1, 2, 3, 4 }, .expected = .{ 20, 48, 138, 216 } };
const second = Fixture{ .a = .{ 2, 0, 1, 2 }, .b = .{ 1, 3, 2, 1 }, .c = .{ 2, 1, 1, 2 }, .expected = .{ 8, 7, 6, 14 } };

fn expect_result(expected: [4]f32, result: Tensor) !void {
    try std.testing.expectEqual(.f32, result.dtype);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, result.shape.const_slice());
    try std.testing.expectEqualSlices(f32, &expected, result.as_const_slice(f32));
}

fn assert_typed_binding_refusals(callable: *Callable) !void {
    const args = try first.args();
    var wrong = args[0];
    wrong.dtype = .f64;
    try expect_validation_error(error.DTypeMismatch, "call input dtype disagrees with compiled descriptor", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong = try Tensor.host(.f32, &.{ 1, 4 }, .{ .borrow = std.mem.sliceAsBytes(&first.a) });
    try expect_validation_error(error.ShapeMismatch, "call input shape disagrees with compiled descriptor", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong = try Tensor.host(.f32, &.{ 2, 2 }, .{ .borrow = std.mem.sliceAsBytes(first.a[0..3]) });
    try expect_validation_error(error.BufferTooSmall, "call input capacity is below the compiled byte count", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong = args[0];
    wrong.backing.host.dtype = .f64;
    try expect_validation_error(error.HostMetadataMismatch, "call input host metadata disagrees with Tensor metadata", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong = args[0];
    wrong.backing.host.shape = .from_slice(&.{ 1, 4 });
    try expect_validation_error(error.HostMetadataMismatch, "call input host metadata disagrees with Tensor metadata", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong = Tensor.abstract(.f32, &.{ 2, 2 });
    try expect_validation_error(error.UnsupportedBacking, "call input requires host backing", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    // admission rejects these tags before reading their payloads
    wrong.backing = .{ .device = undefined };
    try expect_validation_error(error.UnsupportedBacking, "call input requires host backing", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    wrong.backing = .{ .traced = undefined };
    try expect_validation_error(error.UnsupportedBacking, "call input requires host backing", Callable.call, .{ callable, .{ wrong, args[1], args[2] } });
    try expect_validation_error(error.UnsupportedSignature, "call requires a flat Tensor tuple", Callable.call, .{ callable, .{.{args[0]}} });
    var result = try callable.call(args);
    defer result.deinit();
    try expect_result(first.expected, result);
}

fn nested(_: struct { a: Tensor }) Tensor {
    unreachable;
}
fn non_tensor(_: i32) Tensor {
    unreachable;
}
fn multiple(_: Tensor) struct { Tensor, Tensor } {
    unreachable;
}
fn generic(_: anytype) Tensor {
    unreachable;
}

test "compile and call release ownership at every allocation failure" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, compile(expression, failing.allocator(), std.testing.io, test_specs(.f32), .{}));
    failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const callable = try compile(expression, failing.allocator(), std.testing.io, test_specs(.f32), .{});
    const compile_allocations = failing.alloc_index;
    var destroyed = false;
    defer if (!destroyed) callable.deinit();
    try check_binding_allocation_failures(callable);
    failing.fail_index = failing.alloc_index;
    try std.testing.expectError(error.OutOfMemory, callable.call(try first.args()));
    failing.fail_index = std.math.maxInt(usize);
    const mapping = try std.posix.mmap(null, std.heap.page_size_min, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    @memcpy(mapping[0..16], std.mem.sliceAsBytes(&first.a));
    var mapped = Tensor{
        .dtype = .f32,
        .shape = .from_slice(&.{ 2, 2 }),
        .backing = .{ .host = .{
            .dtype = .f32,
            .shape = .from_slice(&.{ 2, 2 }),
            .backing = .{ .mmap = mapping },
        } },
    };
    defer mapped.deinit();
    try std.process.protectMemory(mapping, .{ .read = true });
    var heap = try Tensor.host(.f32, &.{ 2, 2 }, .{ .alloc = std.testing.allocator });
    defer heap.deinit();
    @memcpy(heap.host_data_mut(), std.mem.sliceAsBytes(&first.b));
    const borrowed = try first.args();
    var result = try callable.call(.{ mapped, heap, borrowed[2] });
    defer result.deinit();
    callable.deinit();
    destroyed = true;
    try expect_result(first.expected, result);
    try std.testing.expectEqualSlices(f32, &first.a, mapped.as_const_slice(f32)[0..4]);
    try std.testing.expectEqualSlices(f32, &first.b, heap.as_const_slice(f32));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &first.a);
    // Fail the last adapter allocation after source, store, image, and mapping
    //  ownership exists, exercising the complete initialization unwind
    var late = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = compile_allocations - 1 });
    try std.testing.expectError(error.OutOfMemory, compile(expression, late.allocator(), std.testing.io, test_specs(.f32), .{}));
    try std.testing.expectEqual(late.allocated_bytes, late.freed_bytes);
}

test host_admitted {
    try std.testing.expect(host_admitted(.linux, .x86_64));
    try std.testing.expect(!host_admitted(.windows, .x86_64));
    try std.testing.expect(!host_admitted(.linux, .aarch64));
    try std.testing.expect(!host_admitted(.freestanding, .wasm32));
}

test signature_supported {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expect(!signature_supported(nested, @TypeOf(.{.{ .a = test_specs(.f32)[0] }})));
    try std.testing.expect(!signature_supported(non_tensor, @TypeOf(.{@as(i32, 0)})));
    try std.testing.expect(!signature_supported(multiple, @TypeOf(.{test_specs(.f32)[0]})));
    try std.testing.expect(!signature_supported(generic, @TypeOf(.{test_specs(.f32)[0]})));
    try expect_validation_error(error.UnsupportedSignature, "compile requires concrete Tensor parameters and one Tensor result", compile, .{ nested, allocator, io, .{.{ .a = test_specs(.f32)[0] }}, comptime trace.Options{} });
    try expect_validation_error(error.UnsupportedSignature, "compile requires concrete Tensor parameters and one Tensor result", compile, .{ non_tensor, allocator, io, .{@as(i32, 0)}, comptime trace.Options{} });
    try expect_validation_error(error.UnsupportedSignature, "compile requires concrete Tensor parameters and one Tensor result", compile, .{ multiple, allocator, io, .{test_specs(.f32)[0]}, comptime trace.Options{} });
    try expect_validation_error(error.UnsupportedSignature, "compile requires concrete Tensor parameters and one Tensor result", compile, .{ generic, allocator, io, .{test_specs(.f32)[0]}, comptime trace.Options{} });
}

test "Callable.inspect keeps the native image, mapping, and entry across two calls" {
    const callable = try compile(expression, std.testing.allocator, std.testing.io, test_specs(.f32), .{ .name = "resident_cpu_expression" });
    defer callable.deinit();
    const before = callable.inspect();
    if (before.run.failure) |failure| return failure.cause();
    try std.testing.expectEqualStrings("resident_cpu_expression", before.source.get_function_by_id(before.function_id).?.name);
    const native = before.native.?;
    const required = zc.record.typed_at(.product_contract, before.db, before.required_contract, .latest).?;
    try std.testing.expect(required.semantic_subject.eql(before.source_subject));
    const offered = zc.record.typed_at(.product_contract, before.db, native.offered_contract, .latest).?;
    try std.testing.expect(offered.semantic_subject.eql(before.source_subject));
    const request = zc.record.typed_at(.pull_request, before.db, before.request, .latest).?;
    try std.testing.expect(request.contract.eql(before.required_contract));
    const product = zc.record.typed_at(.pull_product, before.db, native.selected_product, .latest).?;
    try std.testing.expect(product.contract.eql(native.offered_contract));
    const root = try root_resolution(before.run, before.request);
    try std.testing.expectEqualSlices(Id, &.{native.selected_product}, root.resolution.satisfied.products);
    try assert_numerical_records(before);
    var first_result = try callable.call(try first.args());
    defer first_result.deinit();
    try expect_result(first.expected, first_result);
    var second_result = try callable.call(try second.args());
    defer second_result.deinit();
    try expect_result(second.expected, second_result);
    try expect_result(first.expected, first_result);
    try std.testing.expect(first_result.host_data().ptr != second_result.host_data().ptr);
    const after = callable.inspect().native.?;
    try std.testing.expect(native.image == after.image);
    try std.testing.expectEqual(native.mapping, after.mapping);
    try std.testing.expectEqual(native.entry, after.entry);
}

test "Callable.call validates the whole tuple before copying or entering native code" {
    const callable = try compile(expression, std.testing.allocator, std.testing.io, test_specs(.f32), .{});
    defer callable.deinit();
    try assert_typed_binding_refusals(callable);
    const original = try first.args();
    try expect_validation_error(error.UnsupportedSignature, "call requires a flat Tensor tuple", Callable.call, .{ callable, .{ .a = original[0], .b = original[1], .c = original[2] } });
    try expect_validation_error(error.ArgumentCountMismatch, "call argument count disagrees with compiled inputs", Callable.call, .{ callable, .{original[0]} });
    const bindings = &callable.state.native.bindings;
    const snapshot = try std.testing.allocator.dupe(u8, bindings.inputs[0].bytes);
    defer std.testing.allocator.free(snapshot);
    @memset(bindings.output.bytes, 0xa5);
    const changed = try second.args();
    var invalid = changed[2];
    invalid.dtype = .f64;
    try expect_validation_error(error.DTypeMismatch, "call input dtype disagrees with compiled descriptor", Callable.call, .{ callable, .{ changed[0], changed[1], invalid } });
    try std.testing.expectEqualSlices(u8, snapshot, bindings.inputs[0].bytes);
    for (bindings.output.bytes) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
}

fn write_mxcsr(value: u32) void {
    if (comptime builtin.cpu.arch != .x86_64) return;
    asm volatile ("ldmxcsr %[value]"
        :
        : [value] "m" (value),
        : .{ .memory = true });
}

test "Callable.call rejects changed MXCSR controls before native entry" {
    if (comptime builtin.cpu.arch != .x86_64 or builtin.os.tag != .linux) return error.SkipZigTest;
    const runtime = zc.x86_64.runtime_numerics;
    const saved = try runtime.read_mxcsr();
    defer write_mxcsr(saved);
    write_mxcsr(runtime.required_controls);
    var accounting = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const callable = try compile(expression, accounting.allocator(), std.testing.io, test_specs(.f32), .{});
    defer callable.deinit();
    const rejected = [_]u32{
        runtime.required_controls | (1 << 13),
        runtime.required_controls | (2 << 13),
        runtime.required_controls | (3 << 13),
        runtime.required_controls | (1 << 15),
        runtime.required_controls | (1 << 6),
        runtime.required_controls & ~@as(u32, 1 << 7),
        runtime.required_controls & ~@as(u32, 1 << 8),
        runtime.required_controls & ~@as(u32, 1 << 9),
        runtime.required_controls & ~@as(u32, 1 << 10),
        runtime.required_controls & ~@as(u32, 1 << 11),
        runtime.required_controls & ~@as(u32, 1 << 12),
    };
    const args = try first.args();
    for (rejected) |controls| {
        const allocated = accounting.allocated_bytes;
        const freed = accounting.freed_bytes;
        @memset(callable.state.native.bindings.output.bytes, 0xa5);
        write_mxcsr(controls);
        std.debug.assert(TestDiagnostics.expected == null);
        TestDiagnostics.expected = .{ .cause = error.FloatingPointEnvironmentMismatch, .context = "native CPU invocation rejected" };
        defer TestDiagnostics.expected = null;
        const outcome = callable.call(args);
        const after = try runtime.read_mxcsr();
        write_mxcsr(runtime.required_controls);
        try std.testing.expectError(error.FloatingPointEnvironmentMismatch, outcome);
        try std.testing.expectEqual(@as(usize, 1), TestDiagnostics.expected.?.count);
        TestDiagnostics.expected = null;
        try std.testing.expectEqual(controls, after);
        try std.testing.expect(accounting.allocated_bytes > allocated);
        try std.testing.expectEqual(accounting.allocated_bytes - allocated, accounting.freed_bytes - freed);
        for (callable.state.native.bindings.output.bytes) |byte| try std.testing.expectEqual(@as(u8, 0xa5), byte);
        var result = try callable.call(args);
        defer result.deinit();
        try expect_result(first.expected, result);
    }
    for ([_]u32{ 1, 2, 4, 8, 16, 32, runtime.status_mask }) |flags| {
        write_mxcsr(runtime.required_controls | flags);
        var result = try callable.call(args);
        defer result.deinit();
        try expect_result(first.expected, result);
    }
}

test "compile retains an unprepared callable for an unsupported f64 source" {
    const callable = try compile(expression, std.testing.allocator, std.testing.io, test_specs(.f64), .{});
    defer callable.deinit();
    const view = callable.inspect();
    if (view.run.failure) |failure| return failure.cause();
    try std.testing.expect(callable.state == .unprepared);
    try std.testing.expect(view.native == null);
    const root = try root_resolution(view.run, view.request);
    try std.testing.expect(root.resolution == .unsupported);
    try expect_validation_error(error.UnsupportedExecutable, "terminal CPU resolution is unsupported", Callable.call, .{ callable, try first.args() });
}

test "Callable.repeat_demand returns a caller-owned Run with the same Contract and Request" {
    const callable = try compile(expression, std.testing.allocator, std.testing.io, test_specs(.f32), .{});
    defer callable.deinit();
    const before = callable.inspect();
    var repeated = try callable.repeat_demand();
    defer repeated.run.deinit(std.testing.allocator);
    if (repeated.run.failure) |failure| return failure.cause();
    try std.testing.expect(repeated.demand.contract.contract.id.eql(before.required_contract));
    try std.testing.expect(repeated.demand.request.eql(before.request));
    try std.testing.expectEqual(.resolved, repeated.run.termination);
    try std.testing.expectEqual(@as(usize, 0), repeated.run.attempts);
    const root = try root_resolution(&repeated.run, repeated.demand.request);
    try std.testing.expectEqualSlices(Id, &.{before.native.?.selected_product}, root.resolution.satisfied.products);
}

fn assert_numerical_records(view: anytype) !void {
    const allocator = std.testing.allocator;
    const q = zcf.db.query;
    const Cat = zc.schema.Db.Catalog;
    const semantics_role = zc.schema.find_symbol_at(view.db, "executable-semantics", .latest).?;
    const required = zc.record.typed_at(.product_contract, view.db, view.required_contract, .latest).?;
    var found_semantics = false;
    for (required.clauses) |id| {
        const clause = zc.record.typed_at(.contract_clause, view.db, id, .latest).?;
        if (!clause.role.eql(semantics_role)) continue;
        try std.testing.expect(!found_semantics);
        found_semantics = true;
        const symbol = zc.record.typed_at(.symbol, view.db, clause.payload, .latest).?;
        try std.testing.expectEqualStrings(zc.kernel.program.requirement.exact_scalar_semantics_name, symbol.name);
    }
    try std.testing.expect(found_semantics);
    const terminal_offers = comptime q.checked(Cat, .{
        .param_count = 2,
        .query = .{
            .var_count = 2,
            .body = &.{
                q.atom(Cat, "pull_product", .{ .@"@record" = q.p(0), .value = q.v(0) }, .{}),
                q.atom(Cat, "numerical_terminal_assessment", .{ .@"@record" = q.v(1), .artifact = q.v(0), .resulting_profile = q.p(1) }, .{}),
            },
            .project = &.{q.v(1)},
        },
    });
    const product_derivations = comptime q.checked(Cat, .{
        .param_count = 1,
        .query = .{
            .var_count = 1,
            .body = &.{q.atom(Cat, "pull_derivation", .{ .@"@record" = q.v(0), .product = q.p(0) }, .{})},
            .project = &.{q.v(0)},
        },
    });
    const native = view.native.?;
    var terminals = try q.run_static(view.db, allocator, terminal_offers, .{ .params = &.{ q.param(native.selected_product), q.param(native.offered_contract) } });
    defer terminals.deinit();
    var derivations = try q.run_static(view.db, allocator, product_derivations, .{ .params = &.{q.param(native.selected_product)} });
    defer derivations.deinit();
    try std.testing.expect(terminals.rows.len > 0);
    for (terminals.rows) |row| {
        const terminal_id = row[0].id;
        const terminal = zc.record.typed_at(.numerical_terminal_assessment, view.db, terminal_id, .latest).?;
        try std.testing.expectEqual(.held, terminal.result);
        const input = zc.record.typed_at(.pull_product, view.db, terminal.input_product, .latest).?;
        try std.testing.expect(input.contract.eql(terminal.input_contract));
        const lowering = zc.record.typed_at(.numerical_lowering_assessment, view.db, terminal.lowering_assessment, .latest).?;
        try std.testing.expectEqual(.held, lowering.result);
        try std.testing.expect(lowering.offered_profile.eql(input.contract));
        try std.testing.expect(lowering.artifact.eql(input.value));
        var linked = false;
        for (derivations.rows) |derivation_row| {
            const derivation = zc.record.typed_at(.pull_derivation, view.db, derivation_row[0].id, .latest).?;
            if (zc.record.contains_id(derivation.products, terminal.input_product) and
                zc.record.contains_id(derivation.premises, terminal_id) and
                zc.record.contains_id(derivation.premises, terminal.lowering_assessment))
                linked = true;
        }
        try std.testing.expect(linked);
    }
}

fn identity(value: Tensor) Tensor {
    return value;
}

test "Callable.call shares identity bindings and returns independent results that outlive teardown" {
    var accounting = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const callable = try compile(identity, accounting.allocator(), std.testing.io, .{test_specs(.f32)[0]}, .{});
    var destroyed = false;
    defer if (!destroyed) callable.deinit();
    const before = callable.inspect().native.?;
    const bindings = &callable.state.native.bindings;
    try std.testing.expect(bindings.inputs[0].bytes.ptr == bindings.output.bytes.ptr);
    try std.testing.expectEqual(bindings.inputs[0].bytes.len, bindings.output.bytes.len);
    try check_binding_allocation_failures(callable);
    const a = (try first.args())[0];
    const b = (try second.args())[0];
    accounting.fail_index = accounting.alloc_index;
    try std.testing.expectError(error.OutOfMemory, callable.call(.{a}));
    accounting.fail_index = std.math.maxInt(usize);
    var result_a = try callable.call(.{a});
    defer result_a.deinit();
    var result_b = try callable.call(.{b});
    defer result_b.deinit();
    try expect_result(first.a, result_a);
    try expect_result(second.a, result_b);
    try std.testing.expect(result_a.host_data().ptr != result_b.host_data().ptr);
    try std.testing.expect(result_a.host_data().ptr != a.host_data().ptr);
    try std.testing.expect(result_b.host_data().ptr != b.host_data().ptr);
    try std.testing.expect(result_b.host_data().ptr != bindings.output.bytes.ptr);
    const after = callable.inspect().native.?;
    try std.testing.expect(before.image == after.image);
    try std.testing.expectEqual(before.mapping, after.mapping);
    try std.testing.expectEqual(before.entry, after.entry);
    callable.deinit();
    destroyed = true;
    try expect_result(first.a, result_a);
    try expect_result(second.a, result_b);
    try std.testing.expectEqualSlices(f32, &first.a, a.as_const_slice(f32));
    try std.testing.expectEqualSlices(f32, &second.a, b.as_const_slice(f32));
}

test "Callable.Bindings merges transitive external aliases with class capacity and alignment" {
    const image = zc.x86_64.artifact.Image{
        .input_count = 2,
        .output_count = 1,
        .allocations = &.{},
        .buffers = &.{
            .{ .storage = .{ .external = .{ .inputs = &.{0}, .outputs = &.{0}, .alignment = 16 } } },
            .{ .storage = .{ .external = .{ .inputs = &.{1}, .outputs = &.{0}, .alignment = 64 } } },
        },
        .buffer_bytes = &.{ 4, 8 },
        .code = &.{},
    };
    const Check = struct {
        fn run(allocator: std.mem.Allocator, executable: *const zc.x86_64.artifact.Image) !void {
            const inputs = [_]Descriptor{ try Descriptor.init(.f32, &.{1}), try Descriptor.init(.f32, &.{2}) };
            const bindings = try Bindings.init(allocator, executable, &inputs, inputs[1]);
            defer bindings.deinit(allocator);
            try std.testing.expect(bindings.inputs[0].bytes.ptr == bindings.output.bytes.ptr);
            try std.testing.expect(bindings.inputs[1].bytes.ptr == bindings.output.bytes.ptr);
            try std.testing.expectEqual(@as(usize, 8), bindings.output.bytes.len);
            try std.testing.expectEqual(@as(usize, 8), bindings.inputs[0].bytes.len);
            try std.testing.expectEqual(@as(usize, 8), bindings.inputs[1].bytes.len);
            try std.testing.expectEqual(@as(usize, 0), @intFromPtr(bindings.output.bytes.ptr) % 64);
            var owned_count: usize = 0;
            for (bindings.storage) |storage| if (storage != null) {
                owned_count += 1;
            };
            try std.testing.expectEqual(@as(usize, 1), owned_count);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Check.run, .{&image});
}
