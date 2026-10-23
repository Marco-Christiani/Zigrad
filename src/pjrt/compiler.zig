//! StableHLO compilation for a PJRT client.

const std = @import("std");
const compilation = @import("../compilation.zig");
const stablehlo = @import("../stablehlo.zig");
const client_mod = @import("client.zig");
const Execution = @import("execution.zig").Execution;

const log = std.log.scoped(.@"zg/pjrt_compiler");

/// Native PJRT compiler output.
///
/// The artifact remains associated with the client that compiled it. Loading
///  transfers its native executable into a framework loaded program.
pub const Artifact = struct {
    client: *client_mod.Client,
    loaded: client_mod.LoadedExecutable,

    pub fn deinit(self: *Artifact) void {
        self.client.deinit_executable(&self.loaded);
        self.* = undefined;
    }
};

const CompilerInterface = compilation.Compiler(stablehlo.Artifact, Artifact);

/// StableHLO compiler for one PJRT execution context.
pub const Compiler = struct {
    interface: CompilerInterface = .{ .vtable = &vtable },

    execution: *Execution,
    options: client_mod.CompileOptions = .{},

    const vtable: CompilerInterface.VTable = .{
        .compile = compile,
    };

    fn compile(
        interface: *CompilerInterface,
        input: stablehlo.Artifact,
        _: *compilation.Context,
    ) compilation.Error!Artifact {
        const self: *Compiler = @fieldParentPtr("interface", interface);
        const loaded = self.execution.client.compile(
            &self.execution.device,
            input.bytes,
            input.encoding,
            self.options,
        ) catch |err| {
            log.err("PJRT compilation failed: {s}", .{@errorName(err)});
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.InvalidArgument => error.InvalidInput,
                error.Unimplemented => error.Unsupported,
                error.ResourceExhausted => error.ResourceExhausted,
                error.Unavailable => error.Unavailable,
                else => error.CompilationFailed,
            };
        };
        return .{
            .client = self.execution.client,
            .loaded = loaded,
        };
    }
};
