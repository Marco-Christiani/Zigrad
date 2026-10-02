const std = @import("std");
const device = @import("device");

/// Services available while running compilation operations.
pub const Context = struct {
    /// Allocator available to compilation operations.
    allocator: std.mem.Allocator,

    /// I/O implementation available to compilation operations.
    io: std.Io,

    /// Device selected for operations that resolve device-specific policy.
    device: ?device.Device = null,
};
