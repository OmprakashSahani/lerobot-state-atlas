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
  MAX_SYMBOLIC_FINGER_SEPARATION,
  MIN_SYMBOLIC_FINGER_SEPARATION,
  rawGripperValueToSymbolicSeparation,
} from "@/components/viewer/EndEffectorMarker";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";
import type { QuaternionXyzw } from "@/lib/atlas-schema/types";

interface NamedElementProps {
  children?: ReactNode;
  name?: string;
  position?: [number, number, number];
  quaternion?: QuaternionXyzw;
}

function markerChildren(
  arm: "left" | "right",
  orientationXyzw?: QuaternionXyzw,
  recordedGripperValue?: number,
): ReactElement<NamedElementProps>[] {
  const marker = EndEffectorMarker({
    arm,
    position: [0.2, 0.4, 0.1],
    orientationXyzw,
    recordedGripperValue,
  });
  return Children.toArray(marker.props.children).filter(
    (child): child is ReactElement<NamedElementProps> =>
      isValidElement<NamedElementProps>(child),
  );
}

describe("symbolic end-effector marker", () => {
  it("maps raw values to deterministic bounded monotonic display separation", () => {
    const inputs = [-100, -2, 0, 2, 100];
    const separations = inputs.map(rawGripperValueToSymbolicSeparation);

    expect(inputs.map(rawGripperValueToSymbolicSeparation)).toEqual(
      separations,
    );
    expect(separations.every(Number.isFinite)).toBe(true);
    expect(
      separations.every(
        (value) =>
          value >= MIN_SYMBOLIC_FINGER_SEPARATION &&
          value <= MAX_SYMBOLIC_FINGER_SEPARATION,
      ),
    ).toBe(true);
    for (let index = 1; index < separations.length; index += 1) {
      expect(separations[index]).toBeGreaterThan(separations[index - 1]);
    }
  });

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

  it("requires orientation before rendering symbolic fingers", () => {
    const orientationOnly = markerChildren("left", [0, 0, 0, 1]);
    const gripperOnly = markerChildren("left", undefined, -2);
    const orientedGlyph = orientationOnly.find(
      (child) => child.props.name === "left-tool-orientation-glyph",
    );
    const orientedChildren = Children.toArray(
      orientedGlyph?.props.children,
    ).filter(
      (child): child is ReactElement<NamedElementProps> =>
        isValidElement<NamedElementProps>(child),
    );

    expect(
      orientedChildren.some(
        (child) => child.props.name === "left-tool-symbolic-gripper",
      ),
    ).toBe(false);
    expect(gripperOnly.map((child) => child.props.name)).toEqual([
      "left-tool-center",
    ]);
  });

  it("renders symmetric local-X fingers inside the exact orientation group", () => {
    const quaternion: QuaternionXyzw = [0.5, -0.5, 0.5, 0.5];
    const rawValue = 2.25;
    const description = describeEndEffectorMarker(
      "left",
      quaternion,
      rawValue,
    );
    const children = markerChildren("left", quaternion, rawValue);
    const glyph = children.find(
      (child) => child.props.name === "left-tool-orientation-glyph",
    );
    const glyphChildren = Children.toArray(glyph?.props.children).filter(
      (child): child is ReactElement<NamedElementProps> =>
        isValidElement<NamedElementProps>(child),
    );
    const gripper = glyphChildren.find(
      (child) => child.props.name === "left-tool-symbolic-gripper",
    );
    const fingers = Children.toArray(gripper?.props.children).filter(
      (child): child is ReactElement<NamedElementProps> =>
        isValidElement<NamedElementProps>(child),
    );

    expect(glyph?.props.quaternion).toBe(quaternion);
    expect(description.orientationGlyph?.symbolicGripper?.rawRecordedValue).toBe(
      rawValue,
    );
    expect(description.orientationGlyph?.symbolicGripper?.fingerAxis).toBe(
      "local +Z",
    );
    expect(fingers.map((finger) => finger.props.name)).toEqual([
      "left-tool-symbolic-finger-negative-x",
      "left-tool-symbolic-finger-positive-x",
    ]);
    expect(fingers[0].props.position?.[0]).toBe(
      -fingers[1].props.position![0],
    );
    expect(fingers[0].props.position?.[2]).toBeGreaterThan(0);
    expect(fingers[1].props.position?.[2]).toBeGreaterThan(0);
  });

  it("uses greater symbolic separation for a higher raw value", () => {
    const lower = describeEndEffectorMarker("right", [0, 0, 0, 1], -3);
    const higher = describeEndEffectorMarker("right", [0, 0, 0, 1], 4);

    expect(
      higher.orientationGlyph?.symbolicGripper?.separation,
    ).toBeGreaterThan(
      lower.orientationGlyph?.symbolicGripper?.separation ?? Infinity,
    );
  });
});
