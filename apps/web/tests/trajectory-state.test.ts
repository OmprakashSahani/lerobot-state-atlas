import { describe, expect, it } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v1/manifest.json";
import {
  AtlasDataError,
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";

const trajectoryState = {
  orientation: {
    available: true,
    representation: "unit-quaternion",
    componentOrder: ["x", "y", "z", "w"],
    frame: "canonical-shared-world",
    samplePolicy: "recorded-sample",
  },
  gripper: {
    available: true,
    leftSourceComponent: "left_gripper.pos",
    rightSourceComponent: "right_gripper.pos",
    valueSemantics: "raw-device-specific-unproven",
    physicalJawWidthCalibrated: false,
    polarityEstablished: false,
    visualizationGeometryCalibrated: false,
  },
} as const;

function makeManifest(
  orientationAvailable = true,
  gripperAvailable = true,
) {
  return decodeManifest({
    ...manifestJson,
    schema: { ...manifestJson.schema, minor: 2 },
    trajectoryState: {
      orientation: {
        ...trajectoryState.orientation,
        available: orientationAvailable,
      },
      gripper: {
        ...trajectoryState.gripper,
        available: gripperAvailable,
      },
    },
  });
}

function makePayload(options?: {
  orientation?: boolean;
  gripper?: boolean;
}) {
  const orientation = options?.orientation ?? true;
  const gripper = options?.gripper ?? true;
  return {
    schema: {
      name: "lerobot-state-atlas.browser-data",
      major: 1,
      minor: 2,
    },
    episodes: [
      {
        episodeId: 0,
        frameIndices: [10, 11],
        timestampsSeconds: [0, 0.02],
        leftPositionsXyz: [
          [0, 0.4, 0],
          [0.1, 0.4, 0],
        ],
        rightPositionsXyz: [
          [0, -0.4, 0],
          [0.1, -0.4, 0],
        ],
        ...(orientation
          ? {
              leftOrientationsXyzw: [
                [0, 0, 0, 1],
                [0, 0, Math.SQRT1_2, Math.SQRT1_2],
              ],
              rightOrientationsXyzw: [
                [0, 0, 0, 1],
                [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
              ],
            }
          : {}),
        ...(gripper
          ? {
              leftRecordedGripperValues: [-0.5, 2.25],
              rightRecordedGripperValues: [100, -3],
            }
          : {}),
      },
    ],
  };
}

describe("browser-data v1.2 trajectory state", () => {
  it.each([
    [true, true, "available", "available"],
    [true, false, "available", "unavailable"],
    [false, true, "unavailable", "available"],
    [false, false, "unavailable", "unavailable"],
  ] as const)(
    "decodes independently available capabilities",
    (orientation, gripper, orientationStatus, gripperStatus) => {
      const decoded = decodeTrajectories(
        makePayload({ orientation, gripper }),
        makeManifest(orientation, gripper),
      );

      expect(decoded.orientation.status).toBe(orientationStatus);
      expect(decoded.gripper.status).toBe(gripperStatus);
    },
  );

  it("retains XYZW quaternions and authoritative raw values unchanged", () => {
    const decoded = decodeTrajectories(makePayload(), makeManifest());

    expect(decoded.orientation).toMatchObject({
      status: "available",
      data: {
        episodes: [
          {
            leftOrientationsXyzw: [
              [0, 0, 0, 1],
              [0, 0, Math.SQRT1_2, Math.SQRT1_2],
            ],
          },
        ],
      },
    });
    expect(decoded.gripper).toMatchObject({
      status: "available",
      data: {
        episodes: [
          {
            leftRecordedGripperValues: [-0.5, 2.25],
            rightRecordedGripperValues: [100, -3],
          },
        ],
      },
    });
  });

  it.each([
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        delete (
          payload.episodes[0] as Partial<(typeof payload.episodes)[number]>
        ).rightOrientationsXyzw;
      },
    ],
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftOrientationsXyzw!.pop();
      },
    ],
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftOrientationsXyzw![0] = [0, 0, 1] as never;
      },
    ],
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftOrientationsXyzw![0] = [0, 0, 0, 2];
      },
    ],
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftOrientationsXyzw![0][0] = Number.NaN;
      },
    ],
    [
      "orientation",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftOrientationsXyzw![0][0] =
          Number.POSITIVE_INFINITY;
      },
    ],
    [
      "gripper",
      (payload: ReturnType<typeof makePayload>) => {
        delete (
          payload.episodes[0] as Partial<(typeof payload.episodes)[number]>
        ).rightRecordedGripperValues;
      },
    ],
    [
      "gripper",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftRecordedGripperValues!.pop();
      },
    ],
    [
      "gripper",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftRecordedGripperValues![0] =
          Number.POSITIVE_INFINITY;
      },
    ],
    [
      "gripper",
      (payload: ReturnType<typeof makePayload>) => {
        payload.episodes[0].leftRecordedGripperValues![0] = Number.NaN;
      },
    ],
  ] as const)(
    "degrades only malformed %s data",
    (capability, mutate) => {
      const payload = makePayload();
      mutate(payload);
      const decoded = decodeTrajectories(payload, makeManifest());
      const other = capability === "orientation" ? "gripper" : "orientation";

      expect(decoded[capability].status).toBe("degraded");
      expect(
        decoded[capability].status === "degraded"
          ? decoded[capability].warning
          : "",
      ).not.toBe("");
      expect(decoded[other].status).toBe("available");
      expect(decoded.episodes).toHaveLength(1);
    },
  );

  it("degrades declaration and payload disagreement independently", () => {
    const absent = decodeTrajectories(
      makePayload({ orientation: false, gripper: true }),
      makeManifest(true, true),
    );
    expect(absent.orientation.status).toBe("degraded");
    expect(absent.gripper.status).toBe("available");

    const unexpected = decodeTrajectories(
      makePayload(),
      makeManifest(false, true),
    );
    expect(unexpected.orientation.status).toBe("degraded");
    expect(unexpected.gripper.status).toBe("available");

    const absentGripper = decodeTrajectories(
      makePayload({ orientation: true, gripper: false }),
      makeManifest(true, true),
    );
    expect(absentGripper.gripper.status).toBe("degraded");
    expect(absentGripper.orientation.status).toBe("available");
    expect(absentGripper.episodes).toHaveLength(1);

    const unexpectedGripper = decodeTrajectories(
      makePayload(),
      makeManifest(true, false),
    );
    expect(unexpectedGripper.gripper.status).toBe("degraded");
    expect(unexpectedGripper.orientation.status).toBe("available");
    expect(unexpectedGripper.episodes).toHaveLength(1);
  });

  it("keeps required position playback strict", () => {
    const payload = makePayload();
    payload.episodes[0].rightPositionsXyz.pop();

    expect(() => decodeTrajectories(payload, makeManifest())).toThrow(
      /arrays are inconsistent/,
    );
  });

  it("rejects unsupported future minors", () => {
    expect(() =>
      decodeManifest({
        ...manifestJson,
        schema: { ...manifestJson.schema, minor: 3 },
      }),
    ).toThrow(/Unsupported browser-data minor version/);
  });

  it("requires exact v1.2 manifest presence and constants", () => {
    const missing = {
      ...manifestJson,
      schema: { ...manifestJson.schema, minor: 2 },
    };
    expect(() => decodeManifest(missing)).toThrow(/exactly when trajectories/);

    const extra = structuredClone(missing);
    extra.payloads = extra.payloads.filter(
      (payload) => payload.kind !== "trajectories",
    );
    Object.assign(extra, { trajectoryState });
    expect(() => decodeManifest(extra)).toThrow(/exactly when trajectories/);

    const invalid = structuredClone({
      ...missing,
      trajectoryState,
    });
    (
      invalid.trajectoryState.orientation as unknown as {
        componentOrder: string[];
      }
    ).componentOrder = [
      "w",
      "x",
      "y",
      "z",
    ];
    expect(() => decodeManifest(invalid)).toThrow(
      /orientation metadata is unsupported/,
    );
  });

  it("rejects every unsupported trajectoryState metadata constant", () => {
    type MutableTrajectoryState = {
      orientation: Record<string, unknown>;
      gripper: Record<string, unknown>;
    };
    const mutations: Array<(state: MutableTrajectoryState) => void> = [
      (state) => {
        state.orientation.representation = "matrix";
      },
      (state) => {
        state.orientation.frame = "tool-local";
      },
      (state) => {
        state.orientation.samplePolicy = "interpolated";
      },
      (state) => {
        state.gripper.leftSourceComponent = "left_gripper.other";
      },
      (state) => {
        state.gripper.rightSourceComponent = "right_gripper.other";
      },
      (state) => {
        state.gripper.valueSemantics = "normalized";
      },
      (state) => {
        state.gripper.physicalJawWidthCalibrated = true;
      },
      (state) => {
        state.gripper.polarityEstablished = true;
      },
      (state) => {
        state.gripper.visualizationGeometryCalibrated = true;
      },
    ];

    for (const mutate of mutations) {
      const invalid = structuredClone({
        ...manifestJson,
        schema: { ...manifestJson.schema, minor: 2 },
        trajectoryState,
      }) as unknown as { trajectoryState: MutableTrajectoryState };
      mutate(invalid.trajectoryState);
      expect(() => decodeManifest(invalid)).toThrow(/metadata is unsupported/);
    }
  });

  it.each([0, 1])(
    "rejects v1.2-only fields under schema v1.%i",
    (minor) => {
      expect(() =>
        decodeManifest({
          ...manifestJson,
          schema: { ...manifestJson.schema, minor },
          trajectoryState,
        }),
      ).toThrow(/unsupported fields/);

      const payload = makePayload();
      payload.schema.minor = minor;
      expect(() =>
        decodeTrajectories(
          payload,
          decodeManifest({
            ...manifestJson,
            schema: { ...manifestJson.schema, minor },
          }),
        ),
      ).toThrow(/unsupported fields/);
    },
  );

  it("keeps v1.0 position-only state explicitly unavailable", () => {
    const payload = makePayload({ orientation: false, gripper: false });
    payload.schema.minor = 0;
    const decoded = decodeTrajectories(
      payload,
      decodeManifest(manifestJson),
    );

    expect(decoded.orientation).toEqual({ status: "unavailable" });
    expect(decoded.gripper).toEqual({ status: "unavailable" });
  });

  it("rejects trajectory schema disagreement", () => {
    const payload = makePayload();
    payload.schema.minor = 1;
    expect(() => decodeTrajectories(payload, makeManifest())).toThrow(
      new AtlasDataError(
        "Trajectory payload schema version must match the manifest.",
      ),
    );
  });
});
