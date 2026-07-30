import type {
  EpisodeVideoSource,
  TrajectoryEpisode,
  Vector3,
} from "@/lib/atlas-schema/types";

export interface PlaybackState {
  frame: number;
  playing: boolean;
  speed: number;
  loop: boolean;
}

export const VIDEO_DRIFT_THRESHOLD_SECONDS = 0.1;

export function episodeVideoTime(
  episode: TrajectoryEpisode,
  source: EpisodeVideoSource,
  frame: number,
): number {
  const index = Math.max(
    0,
    Math.min(episode.timestampsSeconds.length - 1, Math.floor(frame)),
  );
  const elapsed =
    episode.timestampsSeconds[index] - episode.timestampsSeconds[0];
  return Math.min(
    source.toTimestampSeconds,
    Math.max(
      source.fromTimestampSeconds,
      source.fromTimestampSeconds + elapsed,
    ),
  );
}

export function shouldSeekEpisodeVideo(
  currentTime: number,
  targetTime: number,
  immediate: boolean,
  thresholdSeconds = VIDEO_DRIFT_THRESHOLD_SECONDS,
): boolean {
  return (
    immediate ||
    !Number.isFinite(currentTime) ||
    Math.abs(currentTime - targetTime) > thresholdSeconds
  );
}

export function advancePlayback(
  state: PlaybackState,
  elapsedSeconds: number,
  fps: number,
  frameCount: number,
): PlaybackState {
  if (!state.playing || frameCount <= 1) return state;
  const next = state.frame + elapsedSeconds * fps * state.speed;
  const last = frameCount - 1;
  if (next < last) return { ...state, frame: next };
  if (state.loop) return { ...state, frame: next % frameCount };
  return { ...state, frame: last, playing: false };
}

export function playbackPositions(
  episode: TrajectoryEpisode,
  frame: number,
): { left: Vector3; right: Vector3; index: number } {
  const index = Math.max(
    0,
    Math.min(episode.frameIndices.length - 1, Math.floor(frame)),
  );
  return {
    left: episode.leftPositionsXyz[index],
    right: episode.rightPositionsXyz[index],
    index,
  };
}

export function formatPlaybackStatus(
  episode: TrajectoryEpisode,
  frame: number,
): string {
  const index = Math.max(
    0,
    Math.min(episode.frameIndices.length - 1, Math.floor(frame)),
  );
  const elapsed =
    episode.timestampsSeconds[index] - episode.timestampsSeconds[0];
  const total =
    episode.timestampsSeconds.at(-1)! - episode.timestampsSeconds[0];
  return (
    `Sample ${index + 1} of ${episode.frameIndices.length}` +
    ` · Frame index ${episode.frameIndices[index]}` +
    ` · ${elapsed.toFixed(2)} / ${total.toFixed(2)} s`
  );
}
