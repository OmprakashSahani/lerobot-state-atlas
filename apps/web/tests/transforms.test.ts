import { describe, expect, it } from "vitest";

import { applyRigidTransform } from "@/lib/coordinates/transforms";

describe("coordinate transforms", () => {
  it("applies Rz(yaw) and translation in canonical world coordinates", () => {
    const transformed = applyRigidTransform([1, 0, 0], {
      translationXyz: [0, 0.4, 0],
      rotationRpy: [0, 0, Math.PI / 2],
    });

    expect(transformed[0]).toBeCloseTo(0);
    expect(transformed[1]).toBeCloseTo(1.4);
    expect(transformed[2]).toBeCloseTo(0);
  });
});
