import {
  Children,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";
import { describe, expect, it } from "vitest";

import {
  describeEndEffectorMarker,
  EndEffectorMarker,
  END_EFFECTOR_MARKER_COLORS,
} from "@/components/viewer/EndEffectorMarker";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";
import type { QuaternionXyzw } from "@/lib/atlas-schema/types";

interface NamedElementProps {
  children?: ReactNode;
  name?: string;
  quaternion?: QuaternionXyzw;
}

function markerChildren(
  arm: "left" | "right",
  orientationXyzw?: QuaternionXyzw,
): ReactElement<NamedElementProps>[] {
  const marker = EndEffectorMarker({
    arm,
    position: [0.2, 0.4, 0.1],
    orientationXyzw,
  });
  return Children.toArray(marker.props.children).filter(
    (child): child is ReactElement<NamedElementProps> =>
      isValidElement<NamedElementProps>(child),
  );
}

describe("symbolic end-effector marker", () => {
  it("retains the sphere-only fallback without orientation", () => {
    const description = describeEndEffectorMarker("left");
    const children = markerChildren("left");

    expect(description.orientationGlyph).toBeUndefined();
    expect(children.map((child) => child.props.name)).toEqual([
      "left-tool-center",
    ]);
  });

  it("applies the exact XYZW tuple to one orientation group", () => {
    const quaternion: QuaternionXyzw = [0.5, -0.5, 0.5, 0.5];
    const description = describeEndEffectorMarker("left", quaternion);
    const children = markerChildren("left", quaternion);
    const glyph = children.find(
      (child) => child.props.name === "left-tool-orientation-glyph",
    );

    expect(description.orientationGlyph?.quaternionXyzw).toBe(quaternion);
    expect(description.orientationGlyph?.quaternionXyzw).toEqual(quaternion);
    expect(glyph?.props.quaternion).toBe(quaternion);
    expect(glyph?.props.quaternion).toEqual([0.5, -0.5, 0.5, 0.5]);
  });

  it("defines local +Z forward with distinguishable local X and Y axes", () => {
    const description = describeEndEffectorMarker("right", [0, 0, 0, 1]);
    const children = markerChildren("right", [0, 0, 0, 1]);
    const glyph = children.find(
      (child) => child.props.name === "right-tool-orientation-glyph",
    );
    const glyphChildren = Children.toArray(glyph?.props.children).filter(
      (child): child is ReactElement<NamedElementProps> =>
        isValidElement<NamedElementProps>(child),
    );

    expect(description.orientationGlyph).toMatchObject({
      forwardAxis: "local +Z",
      crossAxes: ["local +X", "local +Y"],
    });
    expect(glyphChildren.map((child) => child.props.name)).toEqual([
      "right-tool-forward-local-z",
      "right-tool-forward-tip-local-z",
      "right-tool-local-x",
      "right-tool-local-y",
    ]);
  });

  it("retains arm identity colors", () => {
    expect(describeEndEffectorMarker("left").color).toBe(
      END_EFFECTOR_MARKER_COLORS.left,
    );
    expect(describeEndEffectorMarker("right").color).toBe(
      END_EFFECTOR_MARKER_COLORS.right,
    );
    expect(END_EFFECTOR_MARKER_COLORS.left).not.toBe(
      END_EFFECTOR_MARKER_COLORS.right,
    );
  });

  it("keeps quaternion identity unchanged when runtime spacing moves position", () => {
    const quaternion: QuaternionXyzw = [
      0,
      0,
      Math.SQRT1_2,
      Math.SQRT1_2,
    ];
    const description = describeEndEffectorMarker("left", quaternion);
    const movedPosition = applyRuntimeSpacing(
      [0.2, 0.4, 0.1],
      "left",
      1.2,
      0.8,
    );

    expect(movedPosition).toEqual([0.2, 0.6, 0.1]);
    expect(description.orientationGlyph?.quaternionXyzw).toBe(quaternion);
    expect(description.orientationGlyph?.quaternionXyzw).toEqual([
      0,
      0,
      Math.SQRT1_2,
      Math.SQRT1_2,
    ]);
  });

  it("renders a glyph only for orientation-aware playback", () => {
    expect(
      markerChildren("left", [0, 0, Math.SQRT1_2, Math.SQRT1_2]),
    ).toHaveLength(2);
    expect(markerChildren("left")).toHaveLength(1);
  });
});
