import { describe, expect, it } from "vitest";

import { applyRuntimeSpacing, spacingDeltaY } from "@/lib/coordinates/runtimeSpacing";

describe("runtime arm spacing", () => {
  it("translates arms symmetrically from the exported baseline", () => {
    expect(spacingDeltaY("left", 1, 0.8)).toBeCloseTo(0.1);
    expect(spacingDeltaY("right", 1, 0.8)).toBeCloseTo(-0.1);
  });

  it("uses the same transform for any coverage or trajectory point", () => {
    const point: [number, number, number] = [0.2, 0.4, 0.1];
    expect(applyRuntimeSpacing(point, "left", 1, 0.8)).toEqual([
      0.2,
      0.5,
      0.1,
    ]);
  });

  it("restores exported coordinates at manifest spacing", () => {
    expect(applyRuntimeSpacing([1, 2, 3], "right", 0.8, 0.8)).toEqual([
      1, 2, 3,
    ]);
  });
});
