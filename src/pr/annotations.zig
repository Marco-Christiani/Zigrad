//! Annotation contracts stored in PR regions.

const std = @import("std");
const pr = @import("pr.zig");

/// Region requests for function outlining.
pub const outline = struct {
    /// Region annotation requesting PR function outlining.
    pub const annotation_name = "zigrad.outline";

    /// Unit-valued outline request for region builders.
    pub const annotation: pr.Annotation = .{
        .name = annotation_name,
        .value = .unit,
    };

    /// Invalid payloads for the outline annotation contract.
    pub const AnnotationError = error{
        /// The annotation value is not a supported outline request.
        InvalidOutlineAnnotation,
    };

    /// Return whether a region requests outlining.
    pub fn is_requested(region: pr.Region) AnnotationError!bool {
        const found = region.find_annotation(annotation_name) orelse return false;
        return switch (found.value) {
            .unit => true,
            .boolean => |value| value,
            else => error.InvalidOutlineAnnotation,
        };
    }

    test is_requested {
        var region: pr.Region = .{
            .id = 0,
            .name = "requested",
            .annotations = &.{},
            .op_ids = &.{},
        };
        try std.testing.expect(!try is_requested(region));
        region.annotations = &.{annotation};
        try std.testing.expect(try is_requested(region));

        for ([_]bool{ true, false }) |requested| {
            region.annotations = &.{.{
                .name = annotation_name,
                .value = .{ .boolean = requested },
            }};
            try std.testing.expectEqual(requested, try is_requested(region));
        }

        const unsupported = [_]pr.AnnotationValue{
            .{ .integer = 1 },
            .{ .floating_point = 1.5 },
            .{ .string = "outline" },
            .{ .strings = &.{"outline"} },
            .{ .bytes = &.{1} },
        };
        for (unsupported) |value| {
            region.annotations = &.{.{ .name = annotation_name, .value = value }};
            try std.testing.expectError(error.InvalidOutlineAnnotation, is_requested(region));
        }
    }
};

test {
    std.testing.refAllDecls(outline);
}
