import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ComparisonViewer } from "@/components/checkpoint-comparison/ComparisonViewer";
import { ComparisonEntry } from "@/components/checkpoint-comparison/ComparisonEntry";
import * as comparisonLoader from "@/lib/checkpoint-comparison/load";
import { computeVisibleTrajectoryBounds } from "@/lib/checkpoint-comparison/sceneBounds";
import { decodeCheckpointComparison } from "@/lib/checkpoint-comparison/validate";

vi.mock("@/components/checkpoint-comparison/ComparisonScene", () => ({
  ComparisonScene: ({ bounds, fitRequest, step, visibility }: {
    bounds: { center: number[]; extent: number; gridSize: number };
    fitRequest: number;
    step: number;
    visibility: boolean[];
  }) => (
    <div
      data-testid="comparison-scene"
      data-bounds={JSON.stringify(bounds)}
      data-fit-request={fitRequest}
      data-step={step}
      data-visibility={JSON.stringify(visibility)}
    >
      3D scene
    </div>
  ),
}));

afterEach(cleanup);

function fixture(name: string) {
  const root = resolve(process.cwd(), "../../tests/fixtures", name);
  const manifest = JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf8"));
  manifest.dataset = {
    repositoryId: "DreamMachines/actuator_unboxing_4h_diverse",
    revision: "e973df866c80f52884cc68355579043cab828e78",
  };
  return decodeCheckpointComparison(
    manifest,
    JSON.parse(readFileSync(resolve(root, "plans.json"), "utf8")),
  );
}

function rawAvailable() {
  const root = resolve(process.cwd(), "../../tests/fixtures/checkpoint-comparison-v1.1-available");
  const manifest = JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf8"));
  manifest.dataset = {
    repositoryId: "DreamMachines/actuator_unboxing_4h_diverse",
    revision: "e973df866c80f52884cc68355579043cab828e78",
  };
  return {
    manifest,
    plans: JSON.parse(readFileSync(resolve(root, "plans.json"), "utf8")),
  };
}

function setAuthoritativeTime(
  source: ReturnType<typeof rawAvailable>,
  policyIndex: number,
  stepIndex: number,
  value: unknown,
  updateProjection = false,
) {
  source.plans.plans[policyIndex].relativeTimesSeconds[stepIndex] = value;
  if (updateProjection) {
    source.plans.trajectoryProjection.plans[policyIndex].relativeTimesSeconds[stepIndex] = value;
  }
}

async function loadWithRecordedRequests(bundlePath: string) {
  const source = rawAvailable();
  const bytes = new TextEncoder().encode(JSON.stringify(source.plans));
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  source.manifest.payloads[0].byteSize = bytes.byteLength;
  source.manifest.payloads[0].sha256 = Array.from(
    new Uint8Array(hash),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
  const requests: string[] = [];
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    requests.push(url);
    if (url.endsWith("/manifest.json")) {
      return { ok: true, json: async () => source.manifest } as Response;
    }
    return {
      ok: true,
      arrayBuffer: async () => bytes.slice().buffer,
    } as Response;
  });
  try {
    const decoded = await comparisonLoader.loadCheckpointComparison(bundlePath);
    return { decoded, requests };
  } finally {
    fetchSpy.mockRestore();
  }
}

describe("checkpoint comparison v1.1", () => {
  it("validates the available and unavailable immutable fixtures", () => {
    expect(fixture("checkpoint-comparison-v1.1-available").plans.trajectoryProjection.available).toBe(true);
    expect(fixture("checkpoint-comparison-v1.1-unavailable").plans.trajectoryProjection.available).toBe(false);
  });

  it("rejects v1.0 and reversed policy ordering", () => {
    const root = resolve(process.cwd(), "../../tests/fixtures/checkpoint-comparison-v1");
    expect(() => decodeCheckpointComparison(JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf8")), JSON.parse(readFileSync(resolve(root, "plans.json"), "utf8")))).toThrow(/schema 1.1/);
    const data = fixture("checkpoint-comparison-v1.1-available");
    const malformed = structuredClone(data.plans);
    malformed.plans.reverse();
    expect(() => decodeCheckpointComparison(data.manifest, malformed)).toThrow(/invalid order/);
  });

  it("rejects cross-origin and traversal bundle URLs before fetching", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(comparisonLoader.loadCheckpointComparison("//evil.example/bundle")).rejects.toThrow(/same-origin/);
    await expect(comparisonLoader.loadCheckpointComparison("/atlas/../secret")).rejects.toThrow(/same-origin/);
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it.each([
    "/path/to/comparison",
    "/path/to/comparison/",
  ])("loads canonical bundle requests from %s", async (bundlePath) => {
    const { decoded, requests } = await loadWithRecordedRequests(bundlePath);
    expect(decoded.plans.trajectoryProjection.available).toBe(true);
    expect(requests).toEqual([
      "/path/to/comparison/manifest.json",
      "/path/to/comparison/plans.json",
    ]);
  });

  it.each([
    "",
    "/",
    "path/to/comparison",
    "//example.com/comparison",
    "https://example.com/comparison",
    "/path/to/comparison//",
    "/path//to/comparison",
    "/path/./comparison",
    "/path/../comparison",
    "/path/to/comparison?variant=1",
    "/path/to/comparison#details",
    "/path\\to\\comparison",
  ])("rejects unsafe bundle path %s before fetching", async (bundlePath) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(comparisonLoader.loadCheckpointComparison(bundlePath)).rejects.toThrow(
      /safe same-origin absolute path/,
    );
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it.each([
    "/atlas-data/__local-checkpoint-comparison__/available",
    "/atlas-data/__local-checkpoint-comparison__/available/",
  ])("loads the local visual fixture path form %s", async (bundlePath) => {
    const { requests } = await loadWithRecordedRequests(bundlePath);
    expect(requests).toEqual([
      "/atlas-data/__local-checkpoint-comparison__/available/manifest.json",
      "/atlas-data/__local-checkpoint-comparison__/available/plans.json",
    ]);
  });

  it("validates and normalizes the exact pinned dataset identity", () => {
    const source = rawAvailable();
    expect(decodeCheckpointComparison(source.manifest, source.plans).manifest.dataset).toEqual({
      repositoryId: "DreamMachines/actuator_unboxing_4h_diverse",
      revision: "e973df866c80f52884cc68355579043cab828e78",
    });
  });

  it.each([
    ["missing", undefined, /manifest\.dataset must be an object/],
    ["null", null, /manifest\.dataset must be an object/],
    ["array", [], /manifest\.dataset must be an object/],
    ["scalar", "dataset", /manifest\.dataset must be an object/],
    ["missing repository", { revision: "e973df866c80f52884cc68355579043cab828e78" }, /missing fields: repositoryId/],
    ["missing revision", { repositoryId: "DreamMachines\/actuator_unboxing_4h_diverse" }, /missing fields: revision/],
    ["wrong repository", { repositoryId: "wrong/dataset", revision: "e973df866c80f52884cc68355579043cab828e78" }, /repositoryId must equal/],
    ["wrong revision", { repositoryId: "DreamMachines/actuator_unboxing_4h_diverse", revision: "a".repeat(40) }, /revision must equal/],
    ["short revision", { repositoryId: "DreamMachines/actuator_unboxing_4h_diverse", revision: "e973" }, /revision must equal/],
    ["uppercase revision", { repositoryId: "DreamMachines/actuator_unboxing_4h_diverse", revision: "E973DF866C80F52884CC68355579043CAB828E78" }, /revision must equal/],
    ["extra field", { repositoryId: "DreamMachines/actuator_unboxing_4h_diverse", revision: "e973df866c80f52884cc68355579043cab828e78", task: "extra" }, /unexpected fields: task/],
  ])("rejects %s dataset metadata", (_name, dataset, expected) => {
    const source = rawAvailable();
    if (dataset === undefined) delete source.manifest.dataset;
    else source.manifest.dataset = dataset;
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(expected as RegExp);
  });

  it("renders malformed dataset loading as a comparison-data error", async () => {
    window.history.replaceState({}, "", "/checkpoint-comparison?bundle=/invalid");
    const loader = vi.spyOn(comparisonLoader, "loadCheckpointComparison").mockRejectedValue(
      new Error("manifest.dataset must be an object."),
    );
    render(<ComparisonEntry />);
    expect(await screen.findByRole("alert")).toHaveTextContent("manifest.dataset must be an object");
    loader.mockRestore();
    window.history.replaceState({}, "", "/checkpoint-comparison");
  });

  it("accepts exact 50 FPS timestamps and a cadence value within tolerance", () => {
    const exact = rawAvailable();
    exact.plans.plans.forEach((plan: { relativeTimesSeconds: number[] }) => {
      plan.relativeTimesSeconds = Array.from({ length: 50 }, (_, index) => index / 50);
    });
    exact.plans.trajectoryProjection.plans.forEach(
      (plan: { relativeTimesSeconds: number[] }) => {
        plan.relativeTimesSeconds = Array.from(
          { length: 50 },
          (_, index) => index / 50,
        );
      },
    );
    expect(decodeCheckpointComparison(exact.manifest, exact.plans).plans.plans[0].relativeTimesSeconds).toEqual(
      Array.from({ length: 50 }, (_, index) => index / 50),
    );

    const withinTolerance = rawAvailable();
    setAuthoritativeTime(withinTolerance, 1, 27, 27 / 50 + 1e-8, true);
    expect(
      decodeCheckpointComparison(withinTolerance.manifest, withinTolerance.plans).plans.plans[1]
        .relativeTimesSeconds[27],
    ).toBe(27 / 50 + 1e-8);
  });

  it.each([
    ["positive first", 0, 0, 0.001, /\[0\] must equal 0/],
    ["negative first", 0, 0, -0.001, /\[0\] must equal 0/],
    ["repeated", 0, 12, 11 / 50, /\[12\] must be greater than the previous timestamp/],
    ["decreasing", 0, 12, 10 / 50, /\[12\] must be greater than the previous timestamp/],
    ["later out of order", 1, 43, 41 / 50, /\[43\] must be greater than the previous timestamp/],
    ["wrong increasing cadence", 0, 12, 12 / 50 + 0.001, /\[12\] must equal 12 \/ 50 within tolerance/],
    ["outside tolerance", 1, 27, 27 / 50 + 2.1e-8, /\[27\] must equal 27 \/ 50 within tolerance/],
    ["boolean", 0, 12, true, /\[12\] must be finite/],
    ["NaN", 0, 12, Number.NaN, /\[12\] must be finite/],
    ["positive infinity", 0, 12, Number.POSITIVE_INFINITY, /\[12\] must be finite/],
    ["negative infinity", 1, 12, Number.NEGATIVE_INFINITY, /\[12\] must be finite/],
  ])("rejects %s authoritative timestamp", (_name, policy, step, value, expected) => {
    const source = rawAvailable();
    setAuthoritativeTime(source, policy as number, step as number, value);
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(
      expected as RegExp,
    );
  });

  it("rejects an authoritative timestamp array with the wrong length", () => {
    const source = rawAvailable();
    source.plans.plans[0].relativeTimesSeconds.pop();
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(
      /plans\.plans\[0\]\.relativeTimesSeconds must contain exactly 50 items/,
    );
  });

  it("continues to reject a projected timestamp that differs from its authoritative plan", () => {
    const source = rawAvailable();
    source.plans.trajectoryProjection.plans[1].relativeTimesSeconds[27] += 1e-8;
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(
      /trajectoryProjection\.plans\[1\]\.relativeTimesSeconds\[27\] must match the authoritative plan/,
    );
  });

  it.each([
    ["Base π0.5", 0],
    ["Fine-tuned π0.5", 1],
  ])("renders malformed %s timestamps as an error without comparison data", async (_label, policyIndex) => {
    window.history.replaceState({}, "", "/checkpoint-comparison?bundle=/invalid");
    const loader = vi.spyOn(comparisonLoader, "loadCheckpointComparison").mockRejectedValue(
      new Error(
        `plans.plans[${policyIndex}].relativeTimesSeconds[12] must be greater than the previous timestamp.`,
      ),
    );
    render(<ComparisonEntry />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(`plans.plans[${policyIndex}].relativeTimesSeconds[12]`);
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByTestId("comparison-scene")).not.toBeInTheDocument();
    loader.mockRestore();
    window.history.replaceState({}, "", "/checkpoint-comparison");
  });

  it("accepts normalized empty and ordered joint-limit violations", () => {
    const source = rawAvailable();
    source.plans.trajectoryProjection.plans[0].jointLimitViolations = [];
    source.plans.trajectoryProjection.plans[1].jointLimitViolations = [];
    let decoded = decodeCheckpointComparison(source.manifest, source.plans);
    if (!decoded.plans.trajectoryProjection.available) throw new Error("expected projection");
    expect(decoded.plans.trajectoryProjection.plans[0].jointLimitViolations).toEqual([]);
    source.plans.trajectoryProjection.plans[0].jointLimitViolations = [
      { policyId: "base-pi05", stepIndex: 2, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: -2, bound: -1, violationKind: "lower" },
      { policyId: "base-pi05", stepIndex: 2, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" },
      { policyId: "base-pi05", stepIndex: 3, componentName: "right_joint_6.pos", urdfJointName: "joint6", value: 2, bound: 1, violationKind: "upper" },
    ];
    decoded = decodeCheckpointComparison(source.manifest, source.plans);
    if (!decoded.plans.trajectoryProjection.available) throw new Error("expected projection");
    expect(decoded.plans.trajectoryProjection.plans[0].jointLimitViolations).toHaveLength(3);
  });

  it.each([
    ["missing array", (plan: Record<string, unknown>) => delete plan.jointLimitViolations, /jointLimitViolations must be an array/],
    ["null array", (plan: Record<string, unknown>) => { plan.jointLimitViolations = null; }, /jointLimitViolations must be an array/],
    ["non-array", (plan: Record<string, unknown>) => { plan.jointLimitViolations = {}; }, /jointLimitViolations must be an array/],
    ["malformed object", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [null]; }, /\[0\] must be an object/],
    ["missing field", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05" }]; }, /missing fields/],
    ["extra field", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05", stepIndex: 0, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper", extra: true }]; }, /unexpected fields: extra/],
    ["policy mismatch", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "fine-tuned-pi05", stepIndex: 0, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" }]; }, /policyId must equal base-pi05/],
    ["gripper component", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05", stepIndex: 0, componentName: "left_gripper.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" }]; }, /canonical arm-joint component/],
    ["unknown component", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05", stepIndex: 0, componentName: "joint_7", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" }]; }, /canonical arm-joint component/],
    ["wrong joint", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05", stepIndex: 0, componentName: "left_joint_2.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" }]; }, /urdfJointName must equal joint2/],
    ["invalid kind", (plan: Record<string, unknown>) => { plan.jointLimitViolations = [{ policyId: "base-pi05", stepIndex: 0, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "side" }]; }, /violationKind must be lower or upper/],
  ])("rejects %s violation data", (_name, mutate, expected) => {
    const source = rawAvailable();
    mutate(source.plans.trajectoryProjection.plans[0]);
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(expected as RegExp);
  });

  it.each([true, 1.5, -1, 50])("rejects invalid violation stepIndex %s", (stepIndex) => {
    const source = rawAvailable();
    source.plans.trajectoryProjection.plans[0].jointLimitViolations[0].stepIndex = stepIndex;
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/stepIndex must be an integer from 0 through 49/);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])("rejects nonfinite violation value or bound %s", (invalid) => {
    const source = rawAvailable();
    source.plans.trajectoryProjection.plans[0].jointLimitViolations[0].value = invalid;
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/\.value must be finite/);
    source.plans.trajectoryProjection.plans[0].jointLimitViolations[0].value = 2;
    source.plans.trajectoryProjection.plans[0].jointLimitViolations[0].bound = invalid;
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/\.bound must be finite/);
  });

  it("rejects unsorted, duplicate, and reject-policy violations", () => {
    const violation = { policyId: "base-pi05", stepIndex: 1, componentName: "left_joint_1.pos", urdfJointName: "joint1", value: 2, bound: 1, violationKind: "upper" };
    const source = rawAvailable();
    source.plans.trajectoryProjection.plans[0].jointLimitViolations = [violation, { ...violation, stepIndex: 0 }];
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/not in deterministic order/);
    source.plans.trajectoryProjection.plans[0].jointLimitViolations = [violation, { ...violation }];
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/duplicates/);
    source.plans.trajectoryProjection.plans[0].jointLimitViolations = [violation];
    source.plans.trajectoryProjection.jointLimitPolicy = "reject";
    expect(() => decodeCheckpointComparison(source.manifest, source.plans)).toThrow(/must be empty/);
  });

  it("renders malformed violations as a comparison-data error", async () => {
    window.history.replaceState({}, "", "/checkpoint-comparison?bundle=/invalid");
    const loader = vi.spyOn(comparisonLoader, "loadCheckpointComparison").mockRejectedValue(
      new Error("trajectoryProjection.plans[0].jointLimitViolations must be an array."),
    );
    render(<ComparisonEntry />);
    expect(await screen.findByRole("alert")).toHaveTextContent("jointLimitViolations must be an array");
    loader.mockRestore();
    window.history.replaceState({}, "", "/checkpoint-comparison");
  });

  it("renders generated-action semantics and safety disclaimers", () => {
    render(<ComparisonViewer data={fixture("checkpoint-comparison-v1.1-available")} />);
    expect(screen.getByRole("heading", { name: "Base π0.5 vs Fine-tuned π0.5" })).toBeInTheDocument();
    expect(screen.getByText(/not recorded ground-truth actions/)).toBeInTheDocument();
    expect(screen.getByText(/not physical jaw widths/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("generated joint-limit warning");
    expect(screen.getByLabelText("Comparison step")).toHaveAttribute("max", "49");
    expect(screen.getByRole("button", { name: "Fit view to visible trajectories" })).toBeEnabled();
    expect(
      screen.getByText((_, element) =>
        element?.tagName === "P" &&
        element.textContent?.startsWith("synthetic-v1.1-observation") === true,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/real model inference/i)).not.toBeInTheDocument();
  });

  it("fits the view without changing playback state or the selected step", () => {
    render(<ComparisonViewer data={fixture("checkpoint-comparison-v1.1-available")} />);
    const slider = screen.getByLabelText("Comparison step");
    fireEvent.change(slider, { target: { value: "17" } });
    fireEvent.click(screen.getByRole("button", { name: "Play comparison" }));
    const scene = screen.getByTestId("comparison-scene");
    expect(scene).toHaveAttribute("data-step", "17");
    expect(scene).toHaveAttribute("data-fit-request", "0");
    fireEvent.click(screen.getByRole("button", { name: "Fit view to visible trajectories" }));
    expect(scene).toHaveAttribute("data-step", "17");
    expect(scene).toHaveAttribute("data-fit-request", "1");
    expect(screen.getByRole("button", { name: "Pause comparison" })).toBeInTheDocument();
    expect(slider).toHaveValue("17");
  });

  it("updates the finite bounds source when policy visibility changes", () => {
    const data = fixture("checkpoint-comparison-v1.1-available");
    const projection = data.plans.trajectoryProjection;
    if (!projection.available) throw new Error("expected projection");
    for (const arm of [projection.plans[1].left, projection.plans[1].right]) {
      arm.positionsXyz = arm.positionsXyz.map(() => [10, 0, 0]);
    }
    render(<ComparisonViewer data={data} />);
    const scene = screen.getByTestId("comparison-scene");
    const initial = JSON.parse(scene.getAttribute("data-bounds") ?? "null");
    expect(initial.center.every(Number.isFinite)).toBe(true);
    fireEvent.click(screen.getByLabelText("Fine-tuned π0.5"));
    const baseOnly = JSON.parse(scene.getAttribute("data-bounds") ?? "null");
    expect(baseOnly.center.every(Number.isFinite)).toBe(true);
    expect(baseOnly.center).not.toEqual(initial.center);
    expect(scene).toHaveAttribute("data-visibility", "[true,false]");
  });

  it("uses a safe minimum extent for degenerate and hidden trajectories", () => {
    const data = fixture("checkpoint-comparison-v1.1-available");
    const projection = data.plans.trajectoryProjection;
    if (!projection.available) throw new Error("expected projection");
    for (const plan of projection.plans) {
      for (const arm of [plan.left, plan.right]) {
        arm.positionsXyz = arm.positionsXyz.map(() => [2, 2, 2]);
      }
    }
    const degenerate = computeVisibleTrajectoryBounds(projection, [true, true]);
    const hidden = computeVisibleTrajectoryBounds(projection, [false, false]);
    expect(degenerate.extent).toBe(0.3);
    expect(hidden.extent).toBe(0.3);
    for (const bounds of [degenerate, hidden]) {
      expect([
        ...bounds.center,
        ...bounds.minimum,
        ...bounds.maximum,
        bounds.gridSize,
        bounds.markerRadius,
        bounds.orientationSize,
        bounds.gripperSize,
      ].every(Number.isFinite)).toBe(true);
    }
  });

  it("retains every playback and display control label", () => {
    render(<ComparisonViewer data={fixture("checkpoint-comparison-v1.1-available")} />);
    for (const label of [
      "Base π0.5",
      "Fine-tuned π0.5",
      "Paths",
      "Current markers",
      "XYZW orientation glyphs",
      "Symbolic raw gripper glyphs",
    ]) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
    expect(screen.getByRole("button", { name: "Fit view to visible trajectories" })).toBeInTheDocument();
  });

  it("renders unavailable projection without a 3D scene while retaining actions", () => {
    render(<ComparisonViewer data={fixture("checkpoint-comparison-v1.1-unavailable")} />);
    expect(screen.getByText("No 3D trajectory was fabricated.")).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.queryByTestId("comparison-scene")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Fit view to visible trajectories" })).toBeDisabled();
  });
});
