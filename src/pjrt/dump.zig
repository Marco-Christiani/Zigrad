//! PJRT optimized-program dump operation.

const std = @import("std");

const compilation = @import("compilation");
const LoadedProgram = @import("../execution.zig").LoadedProgram;
const hlo_decode = @import("../c/xla/hlo_decode.zig");
const output = @import("output");
const Execution = @import("execution.zig").Execution;

const log = std.log.scoped(.@"zg/pjrt_dump");

/// Emit the PJRT executable's optimized HLO and return the same loaded program.
pub const DumpOptimizedHlo = struct {
    pub const Input = LoadedProgram;
    pub const Output = LoadedProgram;

    execution: *Execution,
    config: output.Config,

    pub fn run(
        self: DumpOptimizedHlo,
        loaded_program: Input,
        ctx: *compilation.Context,
    ) !Output {
        const loaded = try self.execution.loaded(loaded_program);
        if (try loaded.get_optimized_program(
            self.execution.client.api,
            ctx.allocator,
        )) |optimized| {
            var program = optimized;
            defer program.deinit(ctx.allocator);
            try dump_program(
                ctx.io,
                self.config,
                program.code,
                program.format,
                ctx.allocator,
            );
        } else {
            log.warn("the PJRT client does not expose an optimized program", .{});
        }
        return loaded_program;
    }
};

fn dump_program(
    io: std.Io,
    config: output.Config,
    code: []const u8,
    format: []const u8,
    allocator: std.mem.Allocator,
) !void {
    const task = struct {
        code: []const u8,
        format: []const u8,
        is_stdout: bool,
        allocator: std.mem.Allocator,

        pub fn emit(self: @This(), writer: *std.Io.Writer) !void {
            if (hlo_decode.decode_and_print(self.code, self.allocator, writer)) return;

            if (self.is_stdout) {
                try writer.print("optimized HLO format: {s}\n", .{self.format});
                try writer.print("optimized HLO size: {d} bytes\n", .{self.code.len});
            } else {
                try writer.writeAll(self.code);
            }
        }
    }{
        .code = code,
        .format = format,
        .is_stdout = config.target == .stdout,
        .allocator = allocator,
    };

    try output.write(io, config.target, task);
}
