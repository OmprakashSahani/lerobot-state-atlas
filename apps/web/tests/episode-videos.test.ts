import { describe, expect, it } from "vitest";

import { decodeEpisodeVideos } from "@/lib/atlas-schema/validate";

function makePayload() {
  return {
    schema: {
      name: "lerobot-state-atlas.browser-data",
      major: 1,
      minor: 1,
    },
    defaultCameraId: "top",
    cameras: [
      {
        cameraId: "top",
        datasetFeature: "observation.images.top",
        label: "Top camera",
        width: 224,
        height: 224,
      },
    ],
    episodes: [
      {
        episodeId: 0,
        videos: [
          {
            cameraId: "top",
            filename: "media/episode-000000/top.mp4",
            mimeType: "video/mp4",
            fromTimestampSeconds: 0,
            toTimestampSeconds: 10.3,
            byteSize: 1234,
            sha256: "a".repeat(64),
          },
        ],
      },
    ],
  };
}

describe("episode-video metadata", () => {
  it("decodes valid synchronized video metadata", () => {
    const payload = decodeEpisodeVideos(makePayload());

    expect(payload.defaultCameraId).toBe("top");
    expect(payload.episodes[0].episodeId).toBe(0);
    expect(payload.episodes[0].videos[0].filename).toBe(
      "media/episode-000000/top.mp4",
    );
  });

  it("rejects unsafe media paths", () => {
    for (const filename of [
      "../private.mp4",
      "/private.mp4",
      "C:/private.mp4",
      "media//private.mp4",
      "media/./private.mp4",
      "media\\private.mp4",
      "https://example.com/private.mp4",
      "media/%2e%2e/private.mp4",
      "media/private.mp4?token=secret",
      "media/private.webm",
    ]) {
      const payload = makePayload();
      payload.episodes[0].videos[0].filename = filename;
      expect(() => decodeEpisodeVideos(payload), filename).toThrow(
        /safe bundle-relative MP4 path/,
      );
    }
  });

  it("rejects undeclared cameras and invalid checksums", () => {
    const undeclared = makePayload();
    undeclared.episodes[0].videos[0].cameraId = "left-wrist";
    expect(() => decodeEpisodeVideos(undeclared)).toThrow(
      /undeclared camera/,
    );

    const checksum = makePayload();
    checksum.episodes[0].videos[0].sha256 = "INVALID";
    expect(() => decodeEpisodeVideos(checksum)).toThrow(
      /lowercase SHA-256/,
    );
  });

  it("requires schema v1.1, the declared default camera, and valid bounds", () => {
    const oldSchema = makePayload();
    oldSchema.schema.minor = 0;
    expect(() => decodeEpisodeVideos(oldSchema)).toThrow(/v1.1/);

    const defaultCamera = makePayload();
    defaultCamera.defaultCameraId = "missing";
    expect(() => decodeEpisodeVideos(defaultCamera)).toThrow(
      /declared camera/,
    );

    const missingDefault = makePayload();
    missingDefault.episodes[0].videos[0].cameraId = "other";
    missingDefault.cameras.push({
      ...missingDefault.cameras[0],
      cameraId: "other",
    });
    expect(() => decodeEpisodeVideos(missingDefault)).toThrow();

    const timestamps = makePayload();
    timestamps.episodes[0].videos[0].toTimestampSeconds = 0;
    expect(() => decodeEpisodeVideos(timestamps)).toThrow(/timestamps/);

    const byteSize = makePayload();
    byteSize.episodes[0].videos[0].byteSize = 0;
    expect(() => decodeEpisodeVideos(byteSize)).toThrow(/finite number/);
  });

  it("requires sorted, distinct cameras, sources, episodes, and filenames", () => {
    const duplicateCamera = makePayload();
    duplicateCamera.cameras.push({ ...duplicateCamera.cameras[0] });
    expect(() => decodeEpisodeVideos(duplicateCamera)).toThrow(
      /camera IDs must be unique/,
    );

    const unorderedEpisodes = makePayload();
    unorderedEpisodes.episodes.unshift({
      ...structuredClone(unorderedEpisodes.episodes[0]),
      episodeId: 1,
      videos: [
        {
          ...structuredClone(unorderedEpisodes.episodes[0].videos[0]),
          filename: "media/episode-000001/top.mp4",
        },
      ],
    });
    expect(() => decodeEpisodeVideos(unorderedEpisodes)).toThrow(
      /ordered by episode ID/,
    );

    const duplicateFilename = makePayload();
    duplicateFilename.episodes.push({
      ...structuredClone(duplicateFilename.episodes[0]),
      episodeId: 1,
    });
    expect(() => decodeEpisodeVideos(duplicateFilename)).toThrow(
      /filenames must be globally unique/,
    );
  });

  it("rejects missing and unsupported fields", () => {
    const unsupported = makePayload() as ReturnType<typeof makePayload> & {
      privateUrl?: string;
    };
    unsupported.privateUrl = "https://example.com/video.mp4";
    expect(() => decodeEpisodeVideos(unsupported)).toThrow(
      /unsupported fields/,
    );

    const missing = structuredClone(makePayload()) as Record<string, unknown>;
    delete missing.defaultCameraId;
    expect(() => decodeEpisodeVideos(missing)).toThrow(/missing fields/);
  });
});
