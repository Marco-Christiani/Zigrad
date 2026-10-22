//! Allocator for alignment tests. It honors every requested alignment and
//!  returns misaligned storage for requests below 16 bytes.

const std = @import("std");

/// Allocator that honors every requested alignment and, for requests below 16
///  bytes, returns a pointer that is aligned to the request and to nothing
///  stronger.
pub const AdversarialAllocator = struct {
    parent: std.mem.Allocator,
    records: std.ArrayList(Record) = .empty,
    invalid_frees: usize = 0,
    allocations: usize = 0,
    /// Allocation ordinal, counted from `allocations`, that returns null.
    fail_at: ?usize = null,

    pub const Record = struct {
        ptr: [*]u8,
        len: usize,
        alignment: std.mem.Alignment,
        base: [*]u8,
        base_len: usize,
        frees: usize = 0,
        free_alignment: ?std.mem.Alignment = null,
    };

    const base_alignment: std.mem.Alignment = .@"16";

    pub fn allocator(self: *AdversarialAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{
            .alloc = alloc,
            .resize = std.mem.Allocator.noResize,
            .remap = std.mem.Allocator.noRemap,
            .free = free,
        } };
    }

    pub fn deinit(self: *AdversarialAllocator) void {
        self.records.deinit(self.parent);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *AdversarialAllocator = @ptrCast(@alignCast(ctx));
        defer self.allocations += 1;
        if (self.fail_at == self.allocations) return null;
        const requested = alignment.toByteUnits();
        const offset: usize = if (requested < 16) requested else 0;
        const base_len = len + offset;
        const base = self.parent.rawAlloc(base_len, base_alignment.max(alignment), ret_addr) orelse return null;
        self.records.append(self.parent, .{
            .ptr = base + offset,
            .len = len,
            .alignment = alignment,
            .base = base,
            .base_len = base_len,
        }) catch {
            self.parent.rawFree(base[0..base_len], base_alignment.max(alignment), ret_addr);
            return null;
        };
        return base + offset;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *AdversarialAllocator = @ptrCast(@alignCast(ctx));
        const record = self.find(memory.ptr) orelse {
            self.invalid_frees += 1;
            return;
        };
        record.free_alignment = alignment;
        record.frees += 1;
        if (record.frees != 1 or record.len != memory.len or record.alignment != alignment) {
            self.invalid_frees += 1;
            return;
        }
        self.parent.rawFree(
            record.base[0..record.base_len],
            base_alignment.max(record.alignment),
            ret_addr,
        );
    }

    /// Record for the most recent allocation that starts at `ptr`.
    pub fn find(self: *AdversarialAllocator, ptr: [*]const u8) ?*Record {
        var index = self.records.items.len;
        while (index > 0) {
            index -= 1;
            const record = &self.records.items[index];
            if (record.ptr == ptr) return record;
        }
        return null;
    }

    /// Release count for the allocation that starts at `ptr`. The record
    ///  table can relocate on any allocation, so records are looked up per use.
    pub fn frees_of(self: *AdversarialAllocator, ptr: [*]const u8) usize {
        return self.find(ptr).?.frees;
    }

    pub fn free_alignment_of(self: *AdversarialAllocator, ptr: [*]const u8) ?std.mem.Alignment {
        return self.find(ptr).?.free_alignment;
    }

    /// Fail if any allocation is unreleased or any release was invalid.
    pub fn expect_clean(self: *AdversarialAllocator) !void {
        try std.testing.expectEqual(@as(usize, 0), self.invalid_frees);
        for (self.records.items) |record| try std.testing.expectEqual(@as(usize, 1), record.frees);
    }
};
