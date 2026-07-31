import { describe, expect, it } from "vitest";

import manifestJson from "@/public/atlas-data/demo-v2/manifest.json";
import trajectoriesJson from "@/public/atlas-data/demo-v2/trajectories.json";
import {
  decodeManifest,
  decodeTrajectories,
} from "@/lib/atlas-schema/validate";
import {
  advancePlayback,
  episodeVideoTime,
  formatPlaybackStatus,
  selectRecordedPlaybackSample,
  shouldSeekEpisodeVideo,
  VIDEO_DRIFT_THRESHOLD_SECONDS,
} from "@/lib/playback/controller";
import { applyRuntimeSpacing } from "@/lib/coordinates/runtimeSpacing";
import type {
  TrajectoryEpisode,
  TrajectoryEpisodeOrientations,
  TrajectoryEpisodeRecordedGripperValues,
} from "@/lib/atlas-schema/types";

describe("trajectory playback", () => {
  const manifest = decodeManifest(manifestJson);

  it("validates synchronized committed trajectories", () => {
    const trajectories = decodeTrajectories(trajectoriesJson, manifest);
    const episode = trajectories.episodes[0];
    const orientations =
      trajectories.orientation.status === "available"
        ? trajectories.orientation.data.episodes[0]
        : undefined;
    const gripper =
      trajectories.gripper.status === "available"
        ? trajectories.gripper.data.episodes[0]
        : undefined;
    const sample = selectRecordedPlaybackSample(
      episode,
      2.9,
      orientations,
      gripper,
    );
    expect(sample.index).toBe(2);
    expect(sample.left.position).toEqual(episode.leftPositionsXyz[2]);
    expect(sample.right.position).toEqual(episode.rightPositionsXyz[2]);
    expect(sample.left.orientationXyzw).toEqual(
      orientations?.leftOrientationsXyzw[2],
    );
    expect(sample.left.recordedGripperValue).toBe(
      gripper?.leftRecordedGripperValues[2],
    );
  });

  it("selects one clamped recorded sample for every state field", () => {
    const episode: TrajectoryEpisode = {
      episodeId: 7,
      frameIndices: [10, 11, 12],
      timestampsSeconds: [0, 0.02, 0.04],
      leftPositionsXyz: [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
      ],
      rightPositionsXyz: [
        [-1, 0, 0],
        [-2, 0, 0],
        [-3, 0, 0],
      ],
    };
    const orientations: TrajectoryEpisodeOrientations = {
      episodeId: 7,
      leftOrientationsXyzw: [
        [0, 0, 0, 1],
        [0.1, 0.2, 0.3, 0.9],
        [0, 0, 1, 0],
      ],
      rightOrientationsXyzw: [
        [0, 0, 0, -1],
        [-0.1, -0.2, -0.3, -0.9],
        [0, 1, 0, 0],
      ],
    };
    const gripper: TrajectoryEpisodeRecordedGripperValues = {
      episodeId: 7,
      leftRecordedGripperValues: [-2, 0.5, 4],
      rightRecordedGripperValues: [10, -3, 100],
    };

    const fractional = selectRecordedPlaybackSample(
      episode,
      1.9,
      orientations,
      gripper,
    );
    expect(fractional).toEqual({
      index: 1,
      left: {
        position: [2, 0, 0],
        orientationXyzw: [0.1, 0.2, 0.3, 0.9],
        recordedGripperValue: 0.5,
      },
      right: {
        position: [-2, 0, 0],
        orientationXyzw: [-0.1, -0.2, -0.3, -0.9],
        recordedGripperValue: -3,
      },
    });
    expect(
      selectRecordedPlaybackSample(episode, -10, orientations, gripper).index,
    ).toBe(0);
    const final = selectRecordedPlaybackSample(
      episode,
      100,
      orientations,
      gripper,
    );
    expect(final.index).toBe(2);
    expect(final.left.recordedGripperValue).toBe(4);
    expect(final.right.recordedGripperValue).toBe(100);
  });

  it("supports orientation-only and gripper-only recorded samples", () => {
    const episode: TrajectoryEpisode = {
      episodeId: 2,
      frameIndices: [0],
      timestampsSeconds: [0],
      leftPositionsXyz: [[0, 0.4, 0]],
      rightPositionsXyz: [[0, -0.4, 0]],
    };
    const orientations: TrajectoryEpisodeOrientations = {
      episodeId: 2,
      leftOrientationsXyzw: [[0, 0, 0, 1]],
      rightOrientationsXyzw: [[0, 0, 1, 0]],
    };
    const gripper: TrajectoryEpisodeRecordedGripperValues = {
      episodeId: 2,
      leftRecordedGripperValues: [-0.5],
      rightRecordedGripperValues: [2.5],
    };

    const orientationOnly = selectRecordedPlaybackSample(
      episode,
      0,
      orientations,
    );
    expect(orientationOnly.left.orientationXyzw).toEqual([0, 0, 0, 1]);
    expect(orientationOnly.right.orientationXyzw).toEqual([0, 0, 1, 0]);
    expect(orientationOnly.left.recordedGripperValue).toBeUndefined();

    const gripperOnly = selectRecordedPlaybackSample(
      episode,
      0,
      undefined,
      gripper,
    );
    expect(gripperOnly.left.orientationXyzw).toBeUndefined();
    expect(gripperOnly.left.recordedGripperValue).toBe(-0.5);
    expect(gripperOnly.right.recordedGripperValue).toBe(2.5);
  });

  it("applies runtime spacing only to the selected position", () => {
    const episode: TrajectoryEpisode = {
      episodeId: 3,
      frameIndices: [0],
      timestampsSeconds: [0],
      leftPositionsXyz: [[0.2, 0.4, 0.1]],
      rightPositionsXyz: [[0.2, -0.4, 0.1]],
    };
    const orientations: TrajectoryEpisodeOrientations = {
      episodeId: 3,
      leftOrientationsXyzw: [[0.1, 0.2, 0.3, 0.9]],
      rightOrientationsXyzw: [[-0.1, -0.2, -0.3, -0.9]],
    };
    const sample = selectRecordedPlaybackSample(episode, 0, orientations);
    const originalQuaternion = sample.left.orientationXyzw;

    expect(applyRuntimeSpacing(sample.left.position, "left", 1.2, 0.8)).toEqual(
      [0.2, 0.6, 0.1],
    );
    expect(sample.left.orientationXyzw).toBe(originalQuaternion);
    expect(sample.left.orientationXyzw).toEqual([0.1, 0.2, 0.3, 0.9]);
  });

  it("supports play, pause, FPS timing, speed, and a predictable end", () => {
    const paused = { frame: 2, playing: false, speed: 1, loop: false };
    expect(advancePlayback(paused, 1, 50, 100)).toBe(paused);
    expect(
      advancePlayback({ ...paused, playing: true, speed: 2 }, 0.5, 50, 100).frame,
    ).toBe(52);
    expect(
      advancePlayback({ frame: 98, playing: true, speed: 1, loop: false }, 1, 50, 100),
    ).toEqual({ frame: 99, playing: false, speed: 1, loop: false });
  });

  it("loops predictably", () => {
    expect(
      advancePlayback({ frame: 9, playing: true, speed: 1, loop: true }, 0.1, 10, 10).frame,
    ).toBe(0);
  });

  it("formats first, middle, and final sample status without count ambiguity", () => {
    const episode = decodeTrajectories(trajectoriesJson, manifest).episodes[0];
    expect(formatPlaybackStatus(episode, 0)).toBe(
      "Sample 1 of 515 · Frame index 0 · 0.00 / 10.28 s",
    );
    expect(formatPlaybackStatus(episode, 287.9)).toBe(
      "Sample 288 of 515 · Frame index 287 · 5.74 / 10.28 s",
    );
    expect(formatPlaybackStatus(episode, 514)).toBe(
      "Sample 515 of 515 · Frame index 514 · 10.28 / 10.28 s",
    );
  });

  it("reports Episode 1's exact 445-sample count", () => {
    const episode = decodeTrajectories(trajectoriesJson, manifest).episodes[1];
    expect(formatPlaybackStatus(episode, 444)).toBe(
      "Sample 445 of 445 · Frame index 444 · 8.88 / 8.88 s",
    );
  });

  it("rejects invalid trajectory payloads", () => {
    const invalid = structuredClone(trajectoriesJson);
    invalid.episodes[0].rightPositionsXyz.pop();
    expect(() => decodeTrajectories(invalid, manifest)).toThrow(
      /arrays are inconsistent/,
    );
  });

  it("maps trajectory samples into the bounded video interval", () => {
    const episode = decodeTrajectories(trajectoriesJson, manifest).episodes[0];
    const source = {
      cameraId: "top",
      filename: "media/top.mp4",
      mimeType: "video/mp4" as const,
      fromTimestampSeconds: 4,
      toTimestampSeconds: 4.5,
      byteSize: 1,
      sha256: "a".repeat(64),
    };

    expect(episodeVideoTime(episode, source, 0)).toBe(4);
    expect(episodeVideoTime(episode, source, 5)).toBeCloseTo(
      4 + episode.timestampsSeconds[5] - episode.timestampsSeconds[0],
    );
    expect(episodeVideoTime(episode, source, Number.MAX_SAFE_INTEGER)).toBe(4.5);
  });

  it("corrects video drift only when immediate or beyond the threshold", () => {
    expect(VIDEO_DRIFT_THRESHOLD_SECONDS).toBe(0.1);
    expect(shouldSeekEpisodeVideo(1, 1.05, false)).toBe(false);
    expect(shouldSeekEpisodeVideo(1, 1.11, false)).toBe(true);
    expect(shouldSeekEpisodeVideo(1, 1.01, true)).toBe(true);
  });
});
