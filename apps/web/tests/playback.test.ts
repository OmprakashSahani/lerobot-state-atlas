import { describe, expect, it } from "vitest";

import trajectoriesJson from "@/public/atlas-data/demo-v1/trajectories.json";
import { decodeTrajectories } from "@/lib/atlas-schema/validate";
import {
  advancePlayback,
  episodeVideoTime,
  formatPlaybackStatus,
  playbackPositions,
  shouldSeekEpisodeVideo,
  VIDEO_DRIFT_THRESHOLD_SECONDS,
} from "@/lib/playback/controller";

describe("trajectory playback", () => {
  it("validates synchronized committed trajectories", () => {
    const episode = decodeTrajectories(trajectoriesJson).episodes[0];
    const positions = playbackPositions(episode, 2.9);
    expect(positions.index).toBe(2);
    expect(positions.left).toEqual(episode.leftPositionsXyz[2]);
    expect(positions.right).toEqual(episode.rightPositionsXyz[2]);
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
    const episode = decodeTrajectories(trajectoriesJson).episodes[0];
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
    const episode = decodeTrajectories(trajectoriesJson).episodes[1];
    expect(formatPlaybackStatus(episode, 444)).toBe(
      "Sample 445 of 445 · Frame index 444 · 8.88 / 8.88 s",
    );
  });

  it("rejects invalid trajectory payloads", () => {
    const invalid = structuredClone(trajectoriesJson);
    invalid.episodes[0].rightPositionsXyz.pop();
    expect(() => decodeTrajectories(invalid)).toThrow(/arrays are inconsistent/);
  });

  it("maps trajectory samples into the bounded video interval", () => {
    const episode = decodeTrajectories(trajectoriesJson).episodes[0];
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
