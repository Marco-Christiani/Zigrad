//! Serialized StableHLO artifacts and output operations.

const std = @import("std");
const compilation = @import("compilation");
const output = @import("output");

/// Backend-configuration field carrying an opaque PR custom-call payload.
pub const custom_call_payload_name = "zigrad.payload";

/// StableHLO serialization encoding.
pub const Encoding = enum {
    text,
    binary,
};

/// Serialized StableHLO bytes.
///
/// Release the bytes with `deinit` and the allocator that created them.
pub const Artifact = struct {
    /// Serialized representation.
    bytes: []u8,

    /// Encoding used by `bytes`.
    encoding: Encoding,

    /// Release the serialized representation.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        self.* = undefined;
    }
};

/// Emits textual StableHLO and returns the same artifact.
pub const Dump = struct {
    pub const Input = Artifact;
    pub const Output = Artifact;

    /// Output destination and optional entry label.
    config: output.Config,

    /// Write textual StableHLO and return the artifact unchanged.
    pub fn run(self: Dump, artifact: Input, ctx: *compilation.Context) !Output {
        if (artifact.encoding != .text) return error.TextEncodingRequired;

        const task = struct {
            bytes: []const u8,
            entry: ?[]const u8,

            pub fn emit(task_self: @This(), writer: *std.Io.Writer) !void {
                try write_text(task_self.bytes, task_self.entry, writer);
            }
        }{
            .bytes = artifact.bytes,
            .entry = self.config.entry_label,
        };

        try output.write(ctx.io, self.config.target, task);
        return artifact;
    }
};

fn write_text(bytes: []const u8, entry_label: ?[]const u8, writer: *std.Io.Writer) !void {
    if (entry_label) |name| {
        try writer.print("entry: {s}\n", .{name});
    }
    try writer.writeAll(bytes);
    if (bytes.len == 0 or bytes[bytes.len - 1] != '\n') {
        try writer.writeByte('\n');
    }
}

test write_text {
    var writer_state = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer writer_state.deinit();

    try write_text("module {}", "main", &writer_state.writer);

    const bytes = try writer_state.toOwnedSlice();
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualStrings("entry: main\nmodule {}\n", bytes);
}
