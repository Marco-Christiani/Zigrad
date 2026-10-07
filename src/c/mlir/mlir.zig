//! Zig bindings and helpers for the MLIR C API.
//! Adapted from ZML (https://github.com/zml/zml)
//! Original Copyright (c) 2024 ZML Contributors
//! Apache License 2.0

const std = @import("std");

pub const c = @import("c-mlir");

const log = std.log.scoped(.@"zg/mlir");

test {
    @setEvalBranchQuota(10000);
    std.testing.refAllDecls(@This());

    _ = try Context.init();
}

pub const Error = error{
    /// MLIR rejected or failed to create an object.
    InvalidMlir,
    /// MLIR reported an error with details in the diagnostic log.
    MlirUnexpected,
    /// A requested resource or executor was not found.
    NotFound,
    /// Bytecode version incompatibility.
    InvalidMlirBytecodeVersion,
} || std.mem.Allocator.Error;

pub inline fn string_ref(str: []const u8) c.MlirStringRef {
    return .{ .data = str.ptr, .length = str.len };
}

pub inline fn from_string_ref(str: c.MlirStringRef) []const u8 {
    // MLIR string references do not require null termination.
    return str.data[0..str.length];
}

pub fn register_passes(comptime passes: []const u8) void {
    @field(c, "mlirRegister" ++ passes ++ "Passes")();
}

pub fn success_or(res: c.MlirLogicalResult, err: anytype) @TypeOf(err)!void {
    return if (res.value == 0) err else {};
}

pub const Registry = struct {
    _inner: c.MlirDialectRegistry,

    pub const deinit = helpers.deinit(Registry, c.mlirDialectRegistryDestroy);

    pub fn init() !Registry {
        const registry = c.mlirDialectRegistryCreate();
        return .{ ._inner = .{ .ptr = registry.ptr orelse return Error.MlirUnexpected } };
    }
};

pub const Context = struct {
    _inner: c.MlirContext,
    pub const deinit = helpers.deinit(Context, c.mlirContextDestroy);
    pub const wrap_or = helpers.wrap_or(Context, c.mlirContextIsNull);

    pub fn init() !Context {
        return Context.wrap_or(c.mlirContextCreate()) orelse Error.MlirUnexpected;
    }

    pub fn init_with_registry(registry: Registry, threadingEnabled: bool) !Context {
        return Context.wrap_or(
            c.mlirContextCreateWithRegistry(registry._inner, threadingEnabled),
        ) orelse Error.InvalidMlir;
    }

    pub fn set_multi_threading(self: *Context, enabled: bool) void {
        c.mlirContextEnableMultithreading(self._inner, enabled);
    }

    pub fn append_dialect_registry(self: *Context, registry: Registry) void {
        c.mlirContextAppendDialectRegistry(self._inner, registry._inner);
    }

    pub fn load_all_available_dialects(self: *Context) void {
        c.mlirContextLoadAllAvailableDialects(self._inner);
    }

    pub fn num_registered_dialects(self: Context) usize {
        return @intCast(c.mlirContextGetNumRegisteredDialects(self._inner));
    }

    pub fn num_loaded_dialects(self: Context) usize {
        return @intCast(c.mlirContextGetNumLoadedDialects(self._inner));
    }

    pub fn is_registered_operation(self: Context, op: [:0]const u8) bool {
        return c.mlirContextIsRegisteredOperation(self._inner, string_ref(op));
    }

    pub fn location(self: Context, src: std.lang.SourceLocation) Location {
        return Location.from_src(self, src);
    }

    pub fn allow_unregistered_dialects(self: Context, allow: bool) void {
        c.mlirContextSetAllowUnregisteredDialects(self._inner, allow);
    }
};

pub const Module = struct {
    _inner: c.MlirModule,

    pub const deinit = helpers.deinit(Module, c.mlirModuleDestroy);
    pub const wrap_or = helpers.wrap_or(Module, c.mlirModuleIsNull);

    pub fn init(loc: Location) Module {
        return .{ ._inner = c.mlirModuleCreateEmpty(loc._inner) };
    }

    pub fn parse(ctx: Context, source: [:0]const u8) !Module {
        return Module.wrap_or(
            c.mlirModuleCreateParse(ctx._inner, string_ref(source)),
        ) orelse Error.InvalidMlir;
    }

    /// Parse a module from arbitrary bytes (text or bytecode).
    ///
    /// This accepts `[]const u8` because bytecode may contain null bytes. The
    ///  MLIR C API detects text and bytecode from the input.
    pub fn parse_bytes(ctx: Context, source: []const u8) !Module {
        return Module.wrap_or(
            c.mlirModuleCreateParse(ctx._inner, string_ref(source)),
        ) orelse Error.InvalidMlir;
    }

    pub fn from_operation(operation: Operation) Module {
        return .{ ._inner = c.mlirModuleFromOperation(operation._inner) };
    }

    pub fn context(self: Module) Context {
        return .{ ._inner = c.mlirModuleGetContext(self._inner) };
    }

    pub fn get_body(self: Module) Block {
        return .{ ._inner = c.mlirModuleGetBody(self._inner) };
    }

    pub fn op(self: Module) Operation {
        return .{ ._inner = c.mlirModuleGetOperation(self._inner) };
    }

    pub fn hash(self: Module, hasher: *std.hash.XxHash64) void {
        return self.op().hash(hasher);
    }
};

pub const PassManager = struct {
    _inner: c.MlirPassManager,

    pub const deinit = helpers.deinit(PassManager, c.mlirPassManagerDestroy);
    pub const wrap_or = helpers.wrap_or(PassManager, c.mlirPassManagerIsNull);

    pub fn init(ctx: Context) !PassManager {
        return PassManager.wrap_or(
            c.mlirPassManagerCreate(ctx._inner),
        ) orelse Error.MlirUnexpected;
    }

    pub fn init_on_operation(ctx: Context, op: [:0]const u8) !PassManager {
        return PassManager.wrap_or(
            c.mlirPassManagerCreateOnOperation(ctx._inner, string_ref(op)),
        ) orelse Error.MlirUnexpected;
    }

    pub fn as_op_pass_manager(self: PassManager) OpPassManager {
        return .{ ._inner = c.mlirPassManagerGetAsOpPassManager(self._inner) };
    }

    // TODO(mlir): Expose `mlirPassManagerEnableIRPrinting`.
    // pub fn enableIRPrinting(self: *PassManager) void {}

    pub fn run_on_op(self: *PassManager, op: Operation) error{InvalidMlir}!void {
        if (c.mlirPassManagerRunOnOp(self._inner, op._inner).value == 0) {
            return error.InvalidMlir;
        }
    }
};

fn _mlir_passpipeline_error(err: c.MlirStringRef, ctx: ?*anyopaque) callconv(.c) void {
    _ = err;
    _ = ctx;
}

pub const OpPassManager = struct {
    _inner: c.MlirOpPassManager,

    pub fn add_pipeline(self: *OpPassManager, pipeline: [:0]const u8) error{OutOfMemory}!void {
        if (c.mlirOpPassManagerAddPipeline(
            self._inner,
            string_ref(pipeline),
            &_mlir_passpipeline_error,
            null,
        ).value == 0) {
            return error.OutOfMemory;
        }
    }
};

pub const Identifier = struct {
    _inner: c.MlirIdentifier,

    pub fn get(ctx: Context, str_: [:0]const u8) Identifier {
        return .{ ._inner = c.mlirIdentifierGet(ctx._inner, string_ref(str_)) };
    }

    pub fn context(self: Identifier) Context {
        return .{ ._inner = c.mlirIdentifierGetContext(self._inner) };
    }

    pub fn str(self: Identifier) []const u8 {
        return from_string_ref(c.mlirIdentifierStr(self._inner));
    }

    pub fn equals(self: Identifier, other: Identifier) bool {
        return c.mlirIdentifierEqual(self._inner, other._inner);
    }
};

pub const AttrTuple = struct { [:0]const u8, Attribute };

pub const Attribute = struct {
    _inner: c.MlirAttribute,

    pub const dump = helpers.dump(Attribute, c.mlirAttributeDump);
    pub const eql = helpers.eql(Attribute, c.mlirAttributeEqual);
    pub const format = helpers.format(Attribute, c.mlirAttributePrint);
    pub const wrap_or = helpers.wrap_or(Attribute, c.mlirAttributeIsNull);

    pub fn wrap(c_attr: c.MlirAttribute) Attribute {
        return .{ ._inner = c_attr };
    }

    pub fn parse(ctx: Context, attr: [:0]const u8) !Attribute {
        return Attribute.wrap_or(
            c.mlirAttributeParseGet(ctx._inner, string_ref(attr)),
        ) orelse Error.InvalidMlir;
    }

    pub fn from_any(SpecificAttr: type) fn (x: SpecificAttr) Attribute {
        return struct {
            fn cast(x: SpecificAttr) Attribute {
                return .{ ._inner = x._inner };
            }
        }.cast;
    }

    pub fn is_a(self: Attribute, SpecificAttr: type) bool {
        return SpecificAttr.is_a_fn(self._inner);
    }

    // Attribute constructors erase their concrete wrapper type for composition.
    pub fn null_() Attribute {
        return .wrap(c.mlirAttributeGetNull());
    }

    pub fn string(ctx: Context, str: []const u8) Attribute {
        return StringAttribute.init(ctx, str).as_attr();
    }

    pub fn type_(t: Type) Attribute {
        return TypeAttribute.init(t).as_attr();
    }

    pub fn unit(ctx: Context) Attribute {
        return .wrap(c.mlirUnitAttrGet(ctx._inner));
    }

    pub fn boolean(ctx: Context, value: bool) Attribute {
        return BoolAttribute.init(ctx, value).as_attr();
    }

    pub fn i1_from_bool(ctx: Context, value: bool) Attribute {
        return IntegerAttribute(.i1).init(ctx, @intFromBool(value)).as_attr();
    }

    pub fn int(ctx: Context, comptime int_type: IntegerTypes, value: i64) Attribute {
        return IntegerAttribute(int_type).init(ctx, value).as_attr();
    }

    pub fn float(ctx: Context, comptime ft: FloatTypes, value: f64) Attribute {
        return FloatAttribute(ft).init(ctx, value).as_attr();
    }

    pub fn array(ctx: Context, attrs: []const Attribute) Attribute {
        return ArrayAttribute.init(ctx, attrs).as_attr();
    }

    pub fn dense(ctx: Context, comptime dt: DenseArrayTypes, values: []const dt.ZigType()) Attribute {
        return DenseArrayAttribute(dt).init(ctx, values).as_attr();
    }

    /// Use a tensor as an attribute.
    ///
    /// The tensor is specified by dims, dtype and a flat slice of values.
    pub fn dense_elements(ctx: Context, dims: []const i64, comptime dt: DenseElementsAttributeTypes, values: []const dt.ZigType()) Attribute {
        return DenseElementsAttribute(dt).init(.tensor(dims, dt.mlir_type(ctx)), values).as_attr();
    }

    pub fn dense_elements_from_bytes(ctx: Context, dims: []const i64, dt: DenseElementsAttributeTypes, raw_bytes: []const u8) Attribute {
        const shape: Type = .tensor(dims, dt.mlir_type(ctx));
        return .{ ._inner = c.mlirDenseElementsAttrRawBufferGet(
            shape._inner,
            @intCast(raw_bytes.len),
            raw_bytes.ptr,
        ) };
    }

    pub fn symbol(ctx: Context, flat_name: [:0]const u8) Attribute {
        return FlatSymbolRefAttribute.init(ctx, flat_name).as_attr();
    }

    pub fn named(attr: Attribute, ctx: Context, name: [:0]const u8) NamedAttribute {
        return NamedAttribute.named(ctx, name, attr);
    }

    pub fn dict(ctx: Context, named_attrs: []const AttrTuple) Attribute {
        var attr_buf: [32]NamedAttribute = undefined;
        std.debug.assert(named_attrs.len <= attr_buf.len); // .dict helper only supports up to 32 attributes

        const attrs = attr_buf[0..named_attrs.len];
        for (attrs, named_attrs) |*attr, tuple| {
            attr.* = .named(ctx, tuple[0], tuple[1]);
        }

        return DictionaryAttribute.init(ctx, attrs).as_attr();
    }

    pub fn eql_any(Attr: type) fn (Attr, Attr) bool {
        return struct {
            fn eql(a: Attr, b: Attr) bool {
                return a.as_attr().eql(b.as_attr());
            }
        }.eql;
    }
};

pub const NamedAttribute = extern struct {
    _inner: c.MlirNamedAttribute,

    pub fn wrap(c_named_attribute: c.MlirNamedAttribute) NamedAttribute {
        return @bitCast(c_named_attribute);
    }

    pub fn named(ctx: Context, name: [:0]const u8, attr: Attribute) NamedAttribute {
        return .{ ._inner = .{
            .name = c.mlirIdentifierGet(ctx._inner, string_ref(name)),
            .attribute = attr._inner,
        } };
    }

    pub fn init(name: Identifier, attr: Attribute) NamedAttribute {
        return .{ ._inner = .{
            .name = name._inner,
            .attribute = attr._inner,
        } };
    }
};

pub const StringAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAString;
    pub const as_attr = Attribute.from_any(StringAttribute);
    pub const eql = Attribute.eql_any(StringAttribute);

    pub fn init(ctx: Context, str: []const u8) StringAttribute {
        return .{ ._inner = c.mlirStringAttrGet(ctx._inner, string_ref(str)) };
    }

    pub fn value(self: StringAttribute) []const u8 {
        return from_string_ref(c.mlirStringAttrGetValue(self._inner));
    }
};

pub const BoolAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsABool;
    pub const as_attr = Attribute.from_any(BoolAttribute);
    pub const eql = Attribute.eql_any(BoolAttribute);

    pub fn init(ctx: Context, value_: bool) BoolAttribute {
        return .{ ._inner = c.mlirBoolAttrGet(ctx._inner, if (value_) 1 else 0) };
    }

    pub fn value(self: BoolAttribute) bool {
        return c.mlirBoolAttrGetValue(self._inner);
    }
};

pub const TypeAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAType;
    pub const eql = Attribute.eql_any(TypeAttribute);

    pub fn init(type_: Type) TypeAttribute {
        return .{ ._inner = c.mlirTypeAttrGet(type_._inner) };
    }

    pub fn typ(self: TypeAttribute) Type {
        return .{ ._inner = c.mlirAttributeGetType(self._inner) };
    }

    pub const as_attr = Attribute.from_any(TypeAttribute);
};

pub const ArrayAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAArray;
    pub const as_attr = Attribute.from_any(ArrayAttribute);
    pub const eql = Attribute.eql_any(ArrayAttribute);

    pub fn init(ctx: Context, attrs: []const Attribute) ArrayAttribute {
        return .{ ._inner = c.mlirArrayAttrGet(ctx._inner, @intCast(attrs.len), @ptrCast(attrs.ptr)) };
    }

    pub fn size(self: ArrayAttribute) usize {
        return @intCast(c.mlirArrayAttrGetNumElements(self._inner));
    }

    pub fn get(self: ArrayAttribute, index: usize) Attribute {
        return .{ ._inner = c.mlirArrayAttrGetElement(self._inner, @intCast(index)) };
    }
};

pub fn IntegerAttribute(comptime it: IntegerTypes) type {
    const zig_type, const getter = comptime switch (it) {
        .i1, .i2, .i4, .i8, .i16, .i32, .i64 => .{ i64, c.mlirIntegerAttrGetValueInt },
        .si4, .si8, .si16, .si32, .si64 => .{ i64, c.mlirIntegerAttrGetValueSInt },
        .u2, .u4, .u8, .u16, .u32, .u64 => .{ u64, c.mlirIntegerAttrGetValueUInt },
        .unknown => @compileError("Unsupported integer type"),
    };

    return struct {
        _inner: c.MlirAttribute,
        pub const is_a_fn = c.mlirAttributeIsAInteger;

        pub const IntegerTypeType = IntegerType(it);
        const IntAttr = @This();

        pub const as_attr = Attribute.from_any(IntAttr);
        pub const eql = Attribute.eql_any(IntAttr);

        pub fn init(ctx: Context, value: i64) IntAttr {
            return .{ ._inner = c.mlirIntegerAttrGet(
                IntegerType(it).init(ctx)._inner,
                value,
            ) };
        }

        pub fn get(value: IntAttr) zig_type {
            return @intCast(getter(value._inner));
        }
    };
}

pub fn FloatAttribute(comptime ft: FloatTypes) type {
    return struct {
        _inner: c.MlirAttribute,
        pub const is_a_fn = c.mlirAttributeIsAFloat;
        const FloatAttr = @This();
        pub const as_attr = Attribute.from_any(FloatAttr);

        pub fn init(ctx: Context, value: f64) FloatAttr {
            return .{ ._inner = c.mlirFloatAttrDoubleGet(
                ctx._inner,
                FloatType(ft).init(ctx)._inner,
                value,
            ) };
        }

        pub fn get(value: FloatAttr) f64 {
            return c.mlirFloatAttrGetValueDouble(value._inner);
        }
    };
}

pub const DenseArrayTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,

    pub fn ZigType(comptime dt: DenseArrayTypes) type {
        return switch (dt) {
            .bool => i32,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .f32 => f32,
            .f64 => f64,
        };
    }
};

pub fn DenseArrayAttribute(comptime dt: DenseArrayTypes) type {
    const _is_a_fn, const get_fn, const get_element_fn = switch (dt) {
        .bool => .{ c.mlirAttributeIsADenseBoolArray, c.mlirDenseBoolArrayGet, c.mlirDenseBoolArrayGetElement },
        .i8 => .{ c.mlirAttributeIsADenseI8Array, c.mlirDenseI8ArrayGet, c.mlirDenseI8ArrayGetElement },
        .i16 => .{ c.mlirAttributeIsADenseI16Array, c.mlirDenseI16ArrayGet, c.mlirDenseI16ArrayGetElement },
        .i32 => .{ c.mlirAttributeIsADenseI32Array, c.mlirDenseI32ArrayGet, c.mlirDenseI32ArrayGetElement },
        .i64 => .{ c.mlirAttributeIsADenseI64Array, c.mlirDenseI64ArrayGet, c.mlirDenseI64ArrayGetElement },
        .f32 => .{ c.mlirAttributeIsADenseF32Array, c.mlirDenseF32ArrayGet, c.mlirDenseF32ArrayGetElement },
        .f64 => .{ c.mlirAttributeIsADenseF64Array, c.mlirDenseF64ArrayGet, c.mlirDenseF64ArrayGetElement },
    };

    return struct {
        _inner: c.MlirAttribute,
        const Attr = @This();
        const ElementType = dt;
        const ElementTypeZig = dt.ZigType();

        pub const as_attr = Attribute.from_any(Attr);
        pub const eql = Attribute.eql_any(Attr);
        pub const is_a_fn = _is_a_fn;

        pub fn init(ctx: Context, values: []const ElementTypeZig) Attr {
            return .{ ._inner = get_fn(ctx._inner, @intCast(values.len), @ptrCast(values.ptr)) };
        }

        pub fn get(self: Attr, pos: usize) ElementTypeZig {
            return get_element_fn(self._inner, @intCast(pos));
        }

        pub fn len(self: Attr) usize {
            return @intCast(c.mlirDenseArrayGetNumElements(self._inner));
        }
    };
}

pub const DenseElementsAttributeTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    bf16,
    f16,
    f32,
    f64,
    index,

    pub fn ZigType(comptime dt: DenseElementsAttributeTypes) type {
        return switch (dt) {
            .bool => bool,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .bf16 => u16,
            .f16 => f16,
            .f32 => f32,
            .f64 => f64,
            .index => usize,
        };
    }

    pub fn mlir_type(dt: DenseElementsAttributeTypes, ctx: Context) Type {
        return switch (dt) {
            .bool => .int(ctx, .i1),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .u8 => .int(ctx, .u8),
            .u16 => .int(ctx, .u16),
            .u32 => .int(ctx, .u32),
            .u64 => .int(ctx, .u64),
            .bf16 => .float(ctx, .bf16),
            .f16 => .float(ctx, .f16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .index => .index(ctx),
        };
    }
};

pub fn DenseElementsAttribute(comptime dt: DenseElementsAttributeTypes) type {
    return struct {
        _inner: c.MlirAttribute,

        const Attr = @This();

        pub const is_a_fn = c.mlirAttributeIsADenseElements;
        pub const as_attr = Attribute.from_any(Attr);
        pub const eql = Attribute.eql_any(Attr);

        pub fn init(shaped_type: Type, slice: []const dt.ZigType()) Attr {
            const raw_bytes = std.mem.sliceAsBytes(slice);
            const attr: Attr = .{ ._inner = c.mlirDenseElementsAttrRawBufferGet(
                shaped_type._inner,
                @intCast(raw_bytes.len),
                @ptrCast(raw_bytes.ptr),
            ) };
            std.debug.assert(attr._inner.ptr != null);
            return attr;
        }

        pub fn len(self: Attr) usize {
            return @intCast(c.mlirElementsAttrGetNumElements(self._inner));
        }

        pub fn items(self: Attr) []const dt.ZigType() {
            const raw_bytes: [*]const u8 = c.mlirDenseElementsAttrGetRawData(self._inner) orelse unreachable;
            const ptr: [*]const dt.ZigType() = @ptrCast(@alignCast(raw_bytes));
            // The C API reports an element count, so the element type remains
            //  comptime-known when constructing the slice.
            return ptr[0..self.len()];
        }

        pub fn bytes(self: Attr) []const u8 {
            return std.mem.sliceAsBytes(self.items());
        }
    };
}

pub const FlatSymbolRefAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsAFlatSymbolRef;
    const Self = FlatSymbolRefAttribute;
    pub const eql = Attribute.eql_any(Self);

    pub fn init(ctx: Context, str: [:0]const u8) Self {
        return .{ ._inner = c.mlirFlatSymbolRefAttrGet(ctx._inner, string_ref(str)) };
    }

    pub fn value(self: Self) []const u8 {
        return from_string_ref(c.mlirFlatSymbolRefAttrGetValue(self._inner));
    }

    pub const as_attr = Attribute.from_any(Self);
};

pub const OperationState = struct {
    _inner: c.MlirOperationState,

    const Self = OperationState;

    pub fn init(name: [:0]const u8, loc: Location) Self {
        return .{ ._inner = c.mlirOperationStateGet(string_ref(name), loc._inner) };
    }

    pub fn add_result(self: *Self, type_: Type) void {
        c.mlirOperationStateAddResults(&self._inner, 1, &[_]c.MlirType{type_._inner});
    }

    pub fn add_results(self: *Self, types: []const Type) void {
        c.mlirOperationStateAddResults(&self._inner, @intCast(types.len), @ptrCast(types.ptr));
    }

    pub fn add_operand(self: *Self, value: Value) void {
        c.mlirOperationStateAddOperands(&self._inner, 1, &[_]c.MlirValue{value._inner});
    }

    pub fn add_operands(self: *Self, values: []const Value) void {
        c.mlirOperationStateAddOperands(&self._inner, @intCast(values.len), @ptrCast(values.ptr));
    }

    pub fn add_region(self: *Self, region: *Region) void {
        c.mlirOperationStateAddOwnedRegions(&self._inner, 1, &[_]c.MlirRegion{region._inner});
    }

    pub fn add_regions(self: *Self, regions: []const Region) void {
        c.mlirOperationStateAddOwnedRegions(&self._inner, @intCast(regions.len), @ptrCast(regions.ptr));
    }

    pub fn add_attribute(self: *Self, ctx: Context, name: [:0]const u8, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, 1, @ptrCast(&.{
            .{
                .name = Identifier.get(ctx, name)._inner,
                .attribute = attr._inner,
            },
        }));
    }

    pub fn add_attribute_raw(self: *Self, name: Identifier, attr: Attribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, 1, @ptrCast(&.{
            .{
                .name = name._inner,
                .attribute = attr._inner,
            },
        }));
    }

    pub fn add_attributes(self: *Self, attributes: []const NamedAttribute) void {
        c.mlirOperationStateAddAttributes(&self._inner, @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    pub fn result_type_inference(self: *Self, enabled: bool) void {
        self._inner.enableResultTypeInference = enabled;
    }
};

pub const DictionaryAttribute = struct {
    _inner: c.MlirAttribute,
    pub const is_a_fn = c.mlirAttributeIsADictionary;
    pub const as_attr = Attribute.from_any(DictionaryAttribute);
    pub const eql = Attribute.eql_any(DictionaryAttribute);

    pub fn init(ctx: Context, attributes: []const NamedAttribute) DictionaryAttribute {
        return .{ ._inner = c.mlirDictionaryAttrGet(
            ctx._inner,
            @intCast(attributes.len),
            @ptrCast(attributes.ptr),
        ) };
    }

    pub fn size(self: DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self._inner));
    }

    pub fn get(self: DictionaryAttribute, pos: usize) NamedAttribute {
        return .wrap(c.mlirDictionaryAttrGetElement(self._inner, @bitCast(pos)));
    }

    pub fn get_by_name(self: DictionaryAttribute, name: [:0]const u8) ?Attribute {
        return Attribute.wrap_or(c.mlirDictionaryAttrGetElementByName(self._inner, string_ref(name)));
    }
};

pub const Operation = struct {
    const Self = Operation;
    _inner: c.MlirOperation,

    pub const dump = helpers.dump(Operation, c.mlirOperationDestroy);
    pub const deinit = helpers.deinit(Operation, c.mlirOperationDestroy);
    pub const wrap_or = helpers.wrap_or(Operation, c.mlirOperationIsNull);
    pub const eql = helpers.eql(Operation, c.mlirOperationEqual);

    pub fn init(state: *OperationState) !Self {
        return Self.wrap_or(c.mlirOperationCreate(&state._inner)) orelse Error.InvalidMlir;
    }

    pub fn make(ctx: Context, op_name: [:0]const u8, args: struct {
        operands: ?[]const Value = null,
        variadic_operands: ?[]const []const Value = null,
        tt_variadic_operands: ?[]const []const Value = null,
        results: ?[]const Type = null,
        variadic_results: ?[]const []const Type = null,
        result_type_inference: ?bool = null,
        n_regions: usize = 0,
        attributes: ?[]const AttrTuple = null,
        blocks: ?[]const Block = null,
        verify: bool = true,
        location: Location,
    }) Self {
        var state = OperationState.init(op_name, args.location);
        std.debug.assert(!(args.operands != null and args.variadic_operands != null));
        if (args.operands) |operands| {
            state.add_operands(operands);
        } else if (args.variadic_operands) |operands_segments| {
            const MAX_SEGMENTS = 32;
            var segments_buf: [MAX_SEGMENTS]i32 = undefined;
            var segments_len: usize = 0;

            for (operands_segments) |operands| {
                state.add_operands(operands);
                segments_buf[segments_len] = @intCast(operands.len);
                segments_len += 1;
            }
            state.add_attribute(ctx, "operandSegmentSizes", .dense_elements(ctx, &.{@intCast(segments_len)}, .i32, segments_buf[0..segments_len]));
        } else if (args.tt_variadic_operands) |operands_segments| {
            // Triton expects a dense array for `operandSegmentSizes`.
            const MAX_SEGMENTS = 32;
            var segments_buf: [MAX_SEGMENTS]i32 = undefined;
            var segments_len: usize = 0;

            for (operands_segments) |operands| {
                state.add_operands(operands);
                segments_buf[segments_len] = @intCast(operands.len);
                segments_len += 1;
            }
            state.add_attribute(ctx, "operandSegmentSizes", .dense(ctx, .i32, segments_buf[0..segments_len]));
        }
        if (args.result_type_inference) |enable| {
            state.result_type_inference(enable);
        }
        std.debug.assert(!(args.results != null and args.variadic_results != null));
        if (args.results) |results| {
            state.add_results(results);
        } else if (args.variadic_results) |result_segments| {
            for (result_segments) |results| {
                state.add_results(results);
            }
        }
        for (0..args.n_regions) |_| {
            var region_ = Region.init() catch {
                @panic("Failed to create MLIR region");
            };
            state.add_region(&region_);
        }
        if (args.attributes) |attrs| {
            for (attrs) |attr| {
                state.add_attribute_raw(
                    Identifier.get(ctx, attr[0]),
                    attr[1],
                );
            }
        }
        if (args.blocks) |blocks_| {
            for (blocks_) |block_| {
                var region_ = Region.init() catch {
                    @panic("Failed to create MLIR region");
                };
                region_.append_block(block_);
                state.add_region(&region_);
            }
        }

        const new_op = Operation.init(&state) catch {
            @panic("Failed to create MLIR operation");
        };
        if (args.verify and new_op.verify() == false) {
            log.err("Failed to verify MLIR operation:\n{f}", .{new_op.mlir_formatter(.{ .debug_info = true })});
            @panic("Failed to verify MLIR operation");
        }
        return new_op;
    }

    pub fn init_parse(ctx: Context, str: [:0]const u8) !Self {
        return Self.wrap_or(
            c.mlirOperationCreateParse(ctx._inner, string_ref(str), string_ref("pouet")),
        ) orelse Error.InvalidMlir;
    }

    pub fn clone(self: Self) !Self {
        return Self.wrap_or(
            c.mlirOperationClone(self._inner),
        ) orelse Error.InvalidMlir;
    }

    pub fn name(self: Self) Identifier {
        return .{ ._inner = c.mlirOperationGetName(self._inner) };
    }

    pub fn remove_from_parent(self: *Self) void {
        c.mlirOperationRemoveFromParent(self._inner);
    }

    pub fn num_operands(self: Self) usize {
        return @intCast(c.mlirOperationGetNumOperands(self._inner));
    }

    pub fn operand(self: Self, index: usize) Value {
        return .{ ._inner = c.mlirOperationGetOperand(self._inner, @intCast(index)) };
    }

    pub fn set_operand(self: *Self, index: usize, value: Value) void {
        c.mlirOperationSetOperand(self._inner, @intCast(index), value._inner);
    }

    pub fn num_results(self: Self) usize {
        return @intCast(c.mlirOperationGetNumResults(self._inner));
    }

    pub fn result(self: Self, index: usize) Value {
        return .{ ._inner = c.mlirOperationGetResult(self._inner, @intCast(index)) };
    }

    pub fn next_in_block(self: Self) Self {
        return .{ ._inner = c.mlirOperationGetNextInBlock(self._inner) };
    }

    pub fn block(self: Self) ?Block {
        return Block.wrap_or(c.mlirOperationGetBlock(self._inner));
    }

    pub fn parent(self: Self) ?Self {
        return Self.wrap_or(c.mlirOperationGetParentOperation(self._inner));
    }

    pub fn region(self: Self, index: usize) Region {
        return .{ ._inner = c.mlirOperationGetRegion(self._inner, @intCast(index)) };
    }

    pub fn context(self: Self) Context {
        return .{ ._inner = c.mlirOperationGetContext(self._inner) };
    }

    pub fn write_bytecode(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        var writer_with_err: WriterWithErr = .{ .writer = writer };
        c.mlirOperationWriteBytecode(
            self._inner,
            WriterWithErr.print_callback,
            &writer_with_err,
        );
        return try writer_with_err.check();
    }

    pub fn write_bytecode_with_config(self: Self, writer: *std.Io.Writer, config: struct {
        desiredEmitedVersion: ?i64 = null,
    }) error{ InvalidMlirBytecodeVersion, WriteFailed }!void {
        const cfg = c.mlirBytecodeWriterConfigCreate();
        defer c.mlirBytecodeWriterConfigDestroy(cfg);
        if (config.desiredEmitedVersion) |v| {
            c.mlirBytecodeWriterConfigDesiredEmitVersion(cfg, v);
        }

        var writer_with_err: WriterWithErr = .{ .writer = writer };
        try success_or(c.mlirOperationWriteBytecodeWithConfig(
            self._inner,
            cfg,
            &WriterWithErr.print_callback,
            &writer_with_err,
        ), error.InvalidMlirBytecodeVersion);
        return try writer_with_err.check();
    }

    /// Return a formatter that prints complete MLIR.
    ///
    /// Example: `std.log.debug("{f}", .{module.op().mlir_formatter(.{})});`
    pub fn mlir_formatter(self: Operation, flags: OpPrintingFlags) MlirFormatter {
        return .{ .op = self, .flags = flags };
    }

    const MlirFormatter = struct {
        op: Operation,
        flags: OpPrintingFlags,

        pub fn format(self: MlirFormatter, writer: *std.Io.Writer) !void {
            try self.op.print(writer, self.flags);
        }
    };

    pub fn print(self: Self, writer: *std.Io.Writer, flags: OpPrintingFlags) std.Io.Writer.Error!void {
        const pflags = flags.create();
        defer c.mlirOpPrintingFlagsDestroy(pflags);

        var writer_err: WriterWithErr = .{ .writer = writer };
        c.mlirOperationPrintWithFlags(self._inner, pflags, WriterWithErr.print_callback, &writer_err);
        return try writer_err.check();
    }

    pub fn verify(self: Self) bool {
        return c.mlirOperationVerify(self._inner);
    }

    pub fn get_location(self: Self) Location {
        return .{ ._inner = c.mlirOperationGetLocation(self._inner) };
    }

    pub const WalkOrder = enum(c.MlirWalkOrder) {
        pre_order = c.MlirWalkPreOrder,
        post_order = c.MlirWalkPostOrder,
    };

    pub const WalkResult = enum(c.MlirWalkResult) {
        advance = c.MlirWalkResultAdvance,
        interrupt = c.MlirWalkResultInterrupt,
        skip = c.MlirWalkResultSkip,
    };

    pub fn walk(self: Self, order: WalkOrder, ctx: anytype, walkfn: fn (ctx: anytype, op: Operation) WalkResult) void {
        var inner_ctx = .{ .ctx = ctx };
        const ContextType = @TypeOf(inner_ctx);

        c.mlirOperationWalk(
            self._inner,
            (struct {
                pub fn callback(op: c.MlirOperation, ctx_: ?*anyopaque) callconv(.c) c.MlirWalkResult {
                    const inner_ctx_: *ContextType = @ptrCast(@alignCast(ctx_));
                    return @backingInt(walkfn(inner_ctx_.ctx, .{ ._inner = op }));
                }
            }).callback,
            &inner_ctx,
            @backingInt(order),
        );
    }

    pub fn get_attribute(self: Self, pos: usize) NamedAttribute {
        return .{ ._inner = c.mlirOperationGetAttribute(self._inner, @intCast(pos)) };
    }

    pub fn get_attribute_by_name(self: Self, name_: [:0]const u8) ?Attribute {
        return Attribute.wrap_or(c.mlirOperationGetAttributeByName(self._inner, string_ref(name_)));
    }

    pub fn set_attribute_by_name(self: Self, name_: [:0]const u8, attr: Attribute) void {
        c.mlirOperationSetAttributeByName(self._inner, string_ref(name_), attr._inner);
    }

    pub fn remove_attribute_by_name(self: Self, name_: [:0]const u8) bool {
        return c.mlirOperationRemoveAttributeByName(self._inner, string_ref(name_));
    }

    /// Hash the canonicalized IR, without debug information that can change across builds.
    pub fn hash(op: Operation, hasher: *std.hash.XxHash64) void {
        // Text serialization accepts operations with unused values.
        //
        // MLIR bytecode serialization can fail for those operations. See
        //  https://github.com/zml/zml/issues/97.
        const flags = OpPrintingFlags.create(.{ .debug_info = false });
        defer c.mlirOpPrintingFlagsDestroy(flags);

        c.mlirOperationPrintWithFlags(
            op._inner,
            flags,
            (struct {
                pub fn callback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.c) void {
                    const _hasher: *std.hash.XxHash64 = @ptrCast(@alignCast(ctx_));
                    _hasher.update(str.data[0..str.length]);
                }
            }).callback,
            hasher,
        );
    }
};

pub const OpPrintingFlags = struct {
    elide_large_elements_attrs: ?usize = null,
    debug_info: bool = false,
    debug_info_pretty_form: bool = true,
    print_generic_op_form: bool = false,
    use_local_scope: bool = false,
    assume_verified: bool = false,

    pub fn create(self: OpPrintingFlags) c.MlirOpPrintingFlags {
        const pflags = c.mlirOpPrintingFlagsCreate();
        if (self.elide_large_elements_attrs) |v| {
            c.mlirOpPrintingFlagsElideLargeElementsAttrs(pflags, @intCast(v));
        }
        c.mlirOpPrintingFlagsEnableDebugInfo(pflags, self.debug_info, self.debug_info_pretty_form);
        if (self.print_generic_op_form) {
            c.mlirOpPrintingFlagsPrintGenericOpForm(pflags);
        }
        if (self.use_local_scope) {
            c.mlirOpPrintingFlagsUseLocalScope(pflags);
        }
        if (self.assume_verified) {
            c.mlirOpPrintingFlagsAssumeVerified(pflags);
        }
        return pflags;
    }
};

pub const OpOperand = struct {
    _inner: c.MlirOpOperand,
    pub const wrap_or = helpers.wrap_or(OpOperand, c.mlirOpOperandIsNull);

    pub fn owner(self: OpOperand) Operation {
        return .{ ._inner = c.mlirOpOperandGetOwner(self._inner) };
    }

    pub fn number(self: OpOperand) usize {
        return @intCast(c.mlirOpOperandGetOperandNumber(self._inner));
    }

    pub fn next_use(self: OpOperand) ?OpOperand {
        return wrap_or(c.mlirOpOperandGetNextUse(self._inner));
    }
};

pub const Region = struct {
    _inner: c.MlirRegion,

    pub const eql = helpers.eql(Region, c.mlirRegionEqual);
    pub const deinit = helpers.deinit(Region, c.mlirRegionDestroy);
    pub const wrap_or = helpers.wrap_or(Region, c.mlirRegionIsNull);

    const Self = Region;

    pub fn init() !Self {
        return Self.wrap_or(c.mlirRegionCreate()) orelse Error.InvalidMlir;
    }

    pub fn append_block(self: *Self, block: Block) void {
        c.mlirRegionAppendOwnedBlock(self._inner, block._inner);
    }

    pub fn insert_block(self: *Self, index: isize, block: Block) void {
        c.mlirRegionInsertOwnedBlock(self._inner, index, block._inner);
    }

    pub fn insert_block_before(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockBefore(self._inner, reference._inner, block._inner);
    }

    pub fn insert_block_after(self: *Self, reference: Block, block: Block) void {
        c.mlirRegionInsertOwnedBlockAfter(self._inner, reference._inner, block._inner);
    }

    pub fn first_block(self: Self) Block {
        return .{ ._inner = c.mlirRegionGetFirstBlock(self._inner) };
    }
};

pub const Value = struct {
    _inner: c.MlirValue,

    pub const dump = helpers.dump(Value, c.mlirValueDump);
    pub const eql = helpers.eql(Value, c.mlirValueEqual);
    pub const format = helpers.format(Value, c.mlirValuePrint);
    pub const wrap_or = helpers.wrap_or(Value, c.mlirValueIsNull);

    pub fn get_type(val: Value) Type {
        return .{ ._inner = c.mlirValueGetType(val._inner) };
    }

    pub fn set_type(val: *Value, typ: Type) void {
        c.mlirValueSetType(val._inner, typ._inner);
    }

    pub fn first_use(val: Value) OpOperand {
        return .{ ._inner = c.mlirValueGetFirstUse(val._inner) };
    }

    pub fn replace_all_uses_with(val: Value, with: Value) void {
        c.mlirValueReplaceAllUsesOfWith(val._inner, with._inner);
    }

    pub fn owner(val: Value) Operation {
        return .{ ._inner = c.mlirOpResultGetOwner(val._inner) };
    }

    pub fn is_a_block_argument(val: Value) bool {
        return c.mlirValueIsABlockArgument(val._inner);
    }

    pub fn is_a_op_result(val: Value) bool {
        return c.mlirValueIsAOpResult(val._inner);
    }

    pub const Kind = union(enum) {
        block_argument: BlockArgument,
        op_result: Operation,
        null,
    };

    pub fn kind(val: Value) Kind {
        if (val.is_a_op_result()) {
            return .{ .op_result = val.owner() };
        }
        if (val.is_a_block_argument()) {
            return .{ .block_argument = .{ ._inner = val._inner } };
        }
        // MLIR values are either block arguments or operation results.
        return .null;
    }
};

pub const BlockArgument = struct {
    _inner: c.MlirValue,

    pub fn block(arg: BlockArgument) Block {
        return .{ ._inner = c.mlirBlockArgumentGetOwner(arg._inner) };
    }

    pub fn number(arg: BlockArgument) usize {
        return @bitCast(c.mlirBlockArgumentGetArgNumber(arg._inner));
    }

    pub fn format(self: BlockArgument, writer: *std.Io.Writer) !void {
        const value = Value{ ._inner = self._inner };
        return try value.format(writer);
    }
};

pub const Type = struct {
    _inner: c.MlirType,

    pub const dump = helpers.dump(Type, c.mlirTypeDump);
    pub const eql = helpers.eql(Type, c.mlirTypeEqual);
    pub const format = helpers.format(Type, c.mlirTypePrint);
    pub const wrap_or = helpers.wrap_or(Type, c.mlirTypeIsNull);

    pub fn parse(ctx: Context, str: [:0]const u8) !Type {
        return Type.wrap_or(
            c.mlirTypeParseGet(ctx._inner, string_ref(str)),
        ) orelse Error.InvalidMlir;
    }

    pub fn as(generic: Type, SpecificType: type) ?SpecificType {
        if (@hasDecl(SpecificType, "is_a_fn")) {
            return if (SpecificType.is_a_fn(generic._inner))
                .{ ._inner = generic._inner }
            else
                null;
        }
        @compileError("Mlir subclass of type need `is_a_fn` attribute: " ++ @typeName(SpecificType));
    }

    pub fn from_any(SpecificType: type) fn (x: SpecificType) Type {
        if (!@hasDecl(SpecificType, "as_type")) {
            @compileError("Type.from_any expects a type subclass with 'as_type' declaration");
        }
        return struct {
            fn cast(x: SpecificType) Type {
                return .{ ._inner = x._inner };
            }
        }.cast;
    }

    pub fn eql_any(SpecificType: type) fn (SpecificType, SpecificType) bool {
        return struct {
            fn eql(a: SpecificType, b: SpecificType) bool {
                return a.as_type().eql(b.as_type());
            }
        }.eql;
    }

    pub fn index(ctx: Context) Type {
        return IndexType.init(ctx).as_type();
    }

    pub fn int(ctx: Context, int_type: IntegerTypes) Type {
        return switch (int_type) {
            .unknown => @panic("Unknown integer type"),
            inline else => |t| IntegerType(t).init(ctx).as_type(),
        };
    }

    pub fn float(ctx: Context, ft: FloatTypes) Type {
        return switch (ft) {
            inline else => |t| FloatType(t).init(ctx).as_type(),
        };
    }

    pub fn complex(ctx: Context, ct: ComplexTypes) Type {
        return switch (ct) {
            .unknown => @panic("Unknown complex type can't be created like this"), // What's the point ?
            inline else => |t| ComplexType(t).init(ctx).as_type(),
        };
    }

    pub fn tuple(ctx: Context, types: []const Type) Type {
        return (TupleType.init(ctx, types) catch unreachable).as_type();
    }

    pub fn function(ctx: Context, args: []const Type, results: []const Type) Type {
        return (FunctionType.init(ctx, args, results) catch unreachable).as_type();
    }

    pub fn tensor(dimensions: []const i64, elem_type: Type) Type {
        return RankedTensorType.init(dimensions, elem_type).as_type();
    }
};

pub const IndexType = struct {
    _inner: c.MlirType,

    pub const as_type = Type.from_any(IndexType);
    pub const eql = Type.eql_any(IndexType);
    pub const format = helpers.format(IndexType, c.mlirTypePrint);

    pub fn init(ctx: Context) IndexType {
        return .{ ._inner = c.mlirIndexTypeGet(ctx._inner) };
    }
};

pub const IntegerTypes = enum {
    i1,
    i2,
    i4,
    i8,
    i16,
    i32,
    i64,
    si4,
    si8,
    si16,
    si32,
    si64,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,

    unknown,
};

pub fn IntegerType(comptime it: IntegerTypes) type {
    const Config = switch (it) {
        .i1 => .{ 1, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i2 => .{ 2, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i4 => .{ 4, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i8 => .{ 8, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i16 => .{ 16, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i32 => .{ 32, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .i64 => .{ 64, c.mlirIntegerTypeGet, c.mlirIntegerTypeIsSignless },
        .si4 => .{ 4, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si8 => .{ 8, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si16 => .{ 16, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si32 => .{ 32, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .si64 => .{ 64, c.mlirIntegerTypeSignedGet, c.mlirIntegerTypeIsSigned },
        .u2 => .{ 2, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u4 => .{ 4, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u8 => .{ 8, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u16 => .{ 16, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u32 => .{ 32, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .u64 => .{ 64, c.mlirIntegerTypeUnsignedGet, c.mlirIntegerTypeIsUnsigned },
        .unknown => .{ 0, null, null },
    };

    return struct {
        _inner: c.MlirType,

        const Int = @This();
        pub const is_a_fn = switch (it) {
            .unknown => c.mlirTypeIsAInteger,
            else => type_is_a_integer_exact,
        };

        pub const as_type = Type.from_any(Int);
        pub const eql = Type.eql_any(Int);
        pub const format = helpers.format(Int, c.mlirTypePrint);

        fn type_is_a_integer_exact(typ: c.MlirType) callconv(.c) bool {
            const bit_width = Config[0];
            const is_sign = Config[2];
            return c.mlirTypeIsAInteger(typ) and (c.mlirIntegerTypeGetWidth(typ) == bit_width) and is_sign(typ);
        }

        pub const BitWidth = Config[0];

        pub const init = if (it != .unknown) struct {
            pub fn init(ctx: Context) Int {
                const type_get = Config[1];
                return .{ ._inner = type_get(ctx._inner, BitWidth) };
            }
        }.init else {};
    };
}

pub const FloatTypes = enum {
    f4e2m1fn,
    f8e3m4,
    f8e4m3,
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0fnu,
    bf16,
    f16,
    f32,
    f64,

    pub fn as_type(self: FloatTypes, ctx: Context) Type {
        return switch (self) {
            inline else => |ft| FloatType(ft).init(ctx).as_type(),
        };
    }
};

pub fn FloatType(comptime ft: FloatTypes) type {
    const Config: struct {
        *const fn (c.MlirType) callconv(.c) bool,
        *const fn (c.MlirContext) callconv(.c) c.MlirType,
    } = switch (ft) {
        .f4e2m1fn => .{ c.mlirTypeIsAFloat4E2M1FN, c.mlirFloat4E2M1FNTypeGet },
        .f8e3m4 => .{ c.mlirTypeIsAFloat8E3M4, c.mlirFloat8E3M4TypeGet },
        .f8e4m3 => .{ c.mlirTypeIsAFloat8E4M3, c.mlirFloat8E4M3TypeGet },
        .f8e4m3b11fnuz => .{ c.mlirTypeIsAFloat8E4M3B11FNUZ, c.mlirFloat8E4M3B11FNUZTypeGet },
        .f8e4m3fn => .{ c.mlirTypeIsAFloat8E4M3FN, c.mlirFloat8E4M3FNTypeGet },
        .f8e4m3fnuz => .{ c.mlirTypeIsAFloat8E4M3FNUZ, c.mlirFloat8E4M3FNUZTypeGet },
        .f8e5m2 => .{ c.mlirTypeIsAFloat8E5M2, c.mlirFloat8E5M2TypeGet },
        .f8e5m2fnuz => .{ c.mlirTypeIsAFloat8E5M2FNUZ, c.mlirFloat8E5M2FNUZTypeGet },
        .f8e8m0fnu => .{ c.mlirTypeIsAFloat8E8M0FNU, c.mlirFloat8E8M0FNUTypeGet },
        .bf16 => .{ c.mlirTypeIsABF16, c.mlirBF16TypeGet },
        .f16 => .{ c.mlirTypeIsAF16, c.mlirF16TypeGet },
        .f32 => .{ c.mlirTypeIsAF32, c.mlirF32TypeGet },
        .f64 => .{ c.mlirTypeIsAF64, c.mlirF64TypeGet },
    };

    return struct {
        _inner: c.MlirType,

        const Self = @This();

        pub const is_a_fn = Config[0];

        pub const as_type = Type.from_any(Self);
        pub const eql = Type.eql_any(Self);
        pub const format = helpers.format(Self, c.mlirTypePrint);

        pub fn init(ctx: Context) Self {
            const type_get = Config[1];
            return .{ ._inner = type_get(ctx._inner) };
        }
    };
}

pub const ComplexTypes = enum {
    c64,
    c128,

    unknown,
};

pub fn ComplexType(comptime ct: ComplexTypes) type {
    return struct {
        _inner: c.MlirType,
        const Complex = @This();

        fn mlir_c64_type_get(ctx: c.MlirContext) callconv(.c) c.MlirType {
            return c.mlirComplexTypeGet(c.mlirF32TypeGet(ctx));
        }

        fn mlir_c128_type_get(ctx: c.MlirContext) callconv(.c) c.MlirType {
            return c.mlirComplexTypeGet(c.mlirF64TypeGet(ctx));
        }

        fn mlir_type_is_ac64(typ: c.MlirType) callconv(.c) bool {
            const element_type: c.MlirType = c.mlirComplexTypeGetElementType(typ);
            return c.mlirTypeIsAF32(element_type);
        }

        fn mlir_type_is_ac128(typ: c.MlirType) callconv(.c) bool {
            const element_type: c.MlirType = c.mlirComplexTypeGetElementType(typ);
            return c.mlirTypeIsAF64(element_type);
        }

        const Config = switch (ct) {
            .c64 => .{ mlir_type_is_ac64, mlir_c64_type_get },
            .c128 => .{ mlir_type_is_ac128, mlir_c128_type_get },
            .unknown => .{ c.mlirTypeIsAComplex, null },
        };

        fn type_is_a_unknown_complex(typ: c.MlirType) callconv(.c) bool {
            return c.mlirTypeIsAComplex(typ);
        }

        pub const is_a_fn = Config[0];

        pub const as_type = Type.from_any(Complex);
        pub const eql = Type.eql_any(Complex);
        pub const format = helpers.format(Complex, c.mlirTypePrint);
        pub const ComplexTypeType: ComplexTypes = ct;

        pub const init = if (ct != .unknown) struct {
            pub fn init(ctx: Context) Complex {
                const type_get = Config[1];
                return .{ ._inner = type_get(ctx._inner) };
            }
        }.init else {};
    };
}

pub const TupleType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsATuple;

    pub fn init(ctx: Context, elements: []const Type) !TupleType {
        const tuple_type = c.mlirTupleTypeGet(
            ctx._inner,
            @intCast(elements.len),
            @ptrCast(elements.ptr),
        );
        return .{ ._inner = .{ .ptr = tuple_type.ptr orelse return error.InvalidMlir } };
    }

    pub fn get_num_types(self: TupleType) usize {
        return @intCast(c.mlirTupleTypeGetNumTypes(self._inner));
    }

    pub fn get_element_type(self: TupleType, index: usize) Type {
        return .{ ._inner = c.mlirTupleTypeGetType(self._inner, @intCast(index)) };
    }

    pub const as_type = Type.from_any(TupleType);
};

pub const FunctionType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsAFunction;
    pub const as_type = Type.from_any(FunctionType);
    pub const eql = Type.eql_any(FunctionType);

    pub fn init(ctx: Context, args: []const Type, results: []const Type) !FunctionType {
        const func_type = c.mlirFunctionTypeGet(
            ctx._inner,
            @intCast(args.len),
            @ptrCast(args.ptr),
            @intCast(results.len),
            @ptrCast(results.ptr),
        );
        return .{ ._inner = .{ .ptr = func_type.ptr orelse return error.InvalidMlir } };
    }
};

pub const RankedTensorType = struct {
    _inner: c.MlirType,
    pub const is_a_fn = c.mlirTypeIsARankedTensor;
    pub const as_type = Type.from_any(RankedTensorType);
    pub const eql = Type.eql_any(RankedTensorType);
    pub const format = helpers.format(RankedTensorType, c.mlirTypePrint);

    pub fn init(dimensions: []const i64, elemType: Type) RankedTensorType {
        return .{ ._inner = c.mlirRankedTensorTypeGet(
            @intCast(dimensions.len),
            @ptrCast(dimensions.ptr),
            elemType._inner,
            c.mlirAttributeGetNull(),
        ) };
    }

    pub fn get_element_type(self: RankedTensorType) Type {
        return .{ ._inner = c.mlirShapedTypeGetElementType(self._inner) };
    }

    pub fn get_rank(self: RankedTensorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self._inner));
    }

    pub fn get_dimension(self: RankedTensorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self._inner, @intCast(dim));
    }
};

pub const Dialect = struct {
    _inner: c.MlirDialect,

    pub fn get_context(self: Dialect) Context {
        return .{ ._inner = c.mlirDialectGetContext(self._inner) };
    }

    pub fn get_namespace(self: Dialect) []const u8 {
        return from_string_ref(c.mlirDialectGetNamespace(self._inner));
    }
};

pub const DialectHandle = struct {
    _inner: c.MlirDialectHandle,

    pub fn get_namespace(self: DialectHandle) []const u8 {
        return from_string_ref(c.mlirDialectHandleGetNamespace(self._inner));
    }

    pub fn insert_dialect(self: DialectHandle, registry: Registry) void {
        c.mlirDialectHandleInsertDialect(self._inner, registry._inner);
    }

    pub fn register_dialect(self: DialectHandle, ctx: Context) void {
        c.mlirDialectHandleRegisterDialect(self._inner, ctx._inner);
    }

    pub fn load_dialect(self: DialectHandle, ctx: Context) Dialect {
        return .{ ._inner = c.mlirDialectHandleLoadDialect(self._inner, ctx._inner) };
    }

    pub fn from_string(comptime namespace: []const u8) DialectHandle {
        return .{ ._inner = @field(c, "mlirGetDialectHandle__" ++ namespace ++ "__")() };
    }
};

pub const Location = struct {
    _inner: c.MlirLocation,

    pub const eql = helpers.eql(Location, c.mlirLocationEqual);
    pub const format = helpers.format(Location, c.mlirLocationPrint);

    pub fn from_src(ctx: Context, src: std.lang.SourceLocation) Location {
        return .{ ._inner = c.mlirLocationFileLineColGet(
            ctx._inner,
            string_ref(src.file),
            @intCast(src.line),
            @intCast(src.column),
        ) };
    }

    pub fn file_line_col(ctx: Context, file: []const u8, line: usize, column: usize) Location {
        return .{ ._inner = c.mlirLocationFileLineColGet(
            ctx._inner,
            string_ref(file),
            @intCast(line),
            @intCast(column),
        ) };
    }

    pub fn call_site(callee: Location, caller: Location) Location {
        return .{ ._inner = c.mlirLocationCallSiteGet(callee._inner, caller._inner) };
    }

    pub fn fused(ctx: Context, locations: []const Location, metadata: Attribute) Location {
        return .{ ._inner = c.mlirLocationFusedGet(
            ctx._inner,
            @intCast(locations.len),
            @ptrCast(locations.ptr),
            metadata._inner,
        ) };
    }

    pub fn named(loc: Location, ctx: Context, loc_name: []const u8) Location {
        return .{ ._inner = c.mlirLocationNameGet(ctx._inner, string_ref(loc_name), loc._inner) };
    }

    pub fn named_fmt(loc: Location, ctx: Context, comptime fmt: [:0]const u8, args: anytype) Location {
        var buf: [256]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&buf);
        writer.print(fmt, args) catch {
            buf[256 - 3 ..].* = "...".*;
        };
        return loc.named(ctx, writer.buffered());
    }

    pub fn unknown(ctx: Context) Location {
        return .{ ._inner = c.mlirLocationUnknownGet(ctx._inner) };
    }
};

pub const Block = struct {
    _inner: c.MlirBlock,

    pub const wrap_or = helpers.wrap_or(Block, c.mlirBlockIsNull);
    pub const deinit = helpers.deinit(Block, c.mlirBlockDestroy);
    pub const eql = helpers.eql(Block, c.mlirBlockEqual);

    pub fn init(args: []const Type, locs: []const Location) !Block {
        const block = Block.wrap_or(
            c.mlirBlockCreate(@intCast(args.len), @ptrCast(args.ptr), @ptrCast(locs.ptr)),
        );
        return block orelse error.InvalidMlir;
    }

    pub fn argument(self: Block, index: usize) Value {
        return .{ ._inner = c.mlirBlockGetArgument(self._inner, @intCast(index)) };
    }

    pub fn num_arguments(self: Block) usize {
        return @intCast(c.mlirBlockGetNumArguments(self._inner));
    }

    pub fn add_argument(self: *Block, typ: Type, loc: Location) Value {
        return .{ ._inner = c.mlirBlockAddArgument(self._inner, typ._inner, loc._inner) };
    }

    pub fn insert_argument(self: *Block, index: usize, typ: Type, loc: Location) Value {
        return .{ ._inner = c.mlirBlockInsertArgument(self._inner, @intCast(index), typ._inner, loc._inner) };
    }

    pub fn equals(self: Block, other: Block) bool {
        return c.mlirBlockEqual(self._inner, other._inner);
    }

    pub fn append_operation(self: Block, op: Operation) void {
        c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
    }

    pub fn append_operations(self: *Block, ops: []const Operation) void {
        for (ops) |op| {
            c.mlirBlockAppendOwnedOperation(self._inner, op._inner);
        }
    }

    pub const RecursiveOpts = enum { open, hermetic };

    pub fn append_value_recursive(self: Block, value: Value, opt: RecursiveOpts) void {
        switch (value.kind()) {
            .op_result => |parent_op| self.append_operation_recursive(parent_op, opt),
            .block_argument => |arg| {
                // Hermetic blocks are not allowed to use arguments from other blocks.
                std.debug.assert(opt == .open or self.eql(arg.block()));
            },
            .null => @panic("InvalidMlir"),
        }
    }

    pub fn append_operation_recursive(self: Block, op: Operation, opt: RecursiveOpts) void {
        if (op.block()) |prev_block| {
            // Hermetic blocks are not allowed to reference values from other blocks.
            std.debug.assert(opt == .open or self.equals(prev_block));
            return;
        }
        for (0..op.num_operands()) |i| {
            self.append_value_recursive(op.operand(i), opt);
        }
        self.append_operation(op);
    }
};

pub const helpers = struct {
    pub fn eql(T: type, equal_fn: fn (@FieldType(T, "_inner"), @FieldType(T, "_inner")) callconv(.c) bool) fn (T, T) bool {
        return struct {
            fn eql(a: T, b: T) bool {
                return equal_fn(a._inner, b._inner);
            }
        }.eql;
    }

    pub fn deinit(T: type, deinit_fn: fn (@FieldType(T, "_inner")) callconv(.c) void) fn (*T) void {
        return struct {
            fn deinit(a: *T) void {
                deinit_fn(a._inner);
                a.* = undefined;
            }
        }.deinit;
    }

    pub fn dump(T: type, dump_fn: fn (@FieldType(T, "_inner")) callconv(.c) void) fn (T) void {
        return struct {
            fn dump(a: T) void {
                return dump_fn(a._inner);
            }
        }.dump;
    }

    pub fn is_null(T: type, is_null_fn: fn (@FieldType(T, "_inner")) callconv(.c) bool) fn (T) bool {
        return struct {
            fn is_null(a: T) bool {
                return is_null_fn(a._inner);
            }
        }.is_null;
    }

    pub fn format(
        Any: type,
        print_fn: fn (@FieldType(Any, "_inner"), ?*const MlirStrCallback, ?*anyopaque) callconv(.c) void,
    ) fn (Any, *std.Io.Writer) std.Io.Writer.Error!void {
        return struct {
            pub fn format(self: Any, writer: *std.Io.Writer) std.Io.Writer.Error!void {
                try call_print_fn(Any, self, print_fn, writer);
            }
        }.format;
    }

    pub fn wrap_or(T: type, is_null_fn: fn (@FieldType(T, "_inner")) callconv(.c) bool) fn (@FieldType(T, "_inner")) ?T {
        return struct {
            fn wrap_or(inner: @FieldType(T, "_inner")) ?T {
                if (is_null_fn(inner)) return null;
                return .{ ._inner = inner };
            }
        }.wrap_or;
    }
};

pub const MlirStrCallback = fn (c.MlirStringRef, ?*anyopaque) callconv(.c) void;

pub fn call_print_fn(
    T: type,
    value: T,
    print_fn: fn (@FieldType(T, "_inner"), ?*const MlirStrCallback, ?*anyopaque) callconv(.c) void,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    var writer_with_err: WriterWithErr = .{ .writer = writer };
    print_fn(value._inner, &WriterWithErr.print_callback, &writer_with_err);
    return try writer_with_err.check();
}

pub const WriterWithErr = struct {
    writer: *std.Io.Writer,
    err: ?std.Io.Writer.Error = null,

    pub fn print_callback(mlir_str: c.MlirStringRef, opaque_ctx: ?*anyopaque) callconv(.c) void {
        var ctx: *WriterWithErr = @ptrCast(@alignCast(opaque_ctx));
        if (ctx.err) |_| return;
        ctx.writer.writeAll(from_string_ref(mlir_str)) catch |err| {
            ctx.err = err;
            return;
        };
    }

    pub fn check(self: WriterWithErr) std.Io.Writer.Error!void {
        if (self.err) |err| return err;
    }
};
