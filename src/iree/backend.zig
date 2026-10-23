//! IREE terminal backend composition.

const backend_mod = @import("../backend.zig");
const compilation = @import("../compilation.zig");
const Executor = @import("../execution.zig");
const stablehlo = @import("../stablehlo.zig");
const compiler_mod = @import("compiler.zig");
const config = @import("config.zig");
const execution = @import("execution.zig");
const loader_mod = @import("loader.zig");

const BackendInterface = backend_mod.Backend(stablehlo.Artifact);

/// StableHLO compiler and loader for one IREE execution context.
pub const Backend = struct {
    interface: BackendInterface,
    compiler: compiler_mod.Compiler,
    loader: loader_mod.Loader,

    const vtable: BackendInterface.VTable = .{
        .prepare = prepare,
    };

    pub fn init(
        execution_instance: *execution.Execution,
        compiler_config: config.CompilerConfig,
        entry_name: []const u8,
    ) Backend {
        return .{
            .interface = .{
                .executor = &execution_instance.interface,
                .vtable = &vtable,
            },
            .compiler = .{ .config = compiler_config },
            .loader = .init(execution_instance, entry_name),
        };
    }

    fn prepare(
        interface: *BackendInterface,
        input: stablehlo.Artifact,
        context: *compilation.Context,
    ) backend_mod.Error!Executor.LoadedProgram {
        const self: *Backend = @fieldParentPtr("interface", interface);
        var artifact = try self.compiler.interface.run(input, context);
        errdefer artifact.deinit();
        return try self.loader.interface.load(&artifact);
    }
};
