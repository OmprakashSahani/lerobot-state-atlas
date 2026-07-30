import type {
  AtlasManifest,
  CoverageArm,
  CoveragePayload,
  EpisodeVideoCamera,
  EpisodeVideoEpisode,
  EpisodeVideoPayload,
  EpisodeVideoSource,
  SchemaVersion,
  TrajectoryEpisode,
  TrajectoryPayload,
  Vector3,
} from "./types";

export class AtlasDataError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AtlasDataError";
  }
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new AtlasDataError(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function numberValue(value: unknown, label: string, minimum = 0): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum
  ) {
    throw new AtlasDataError(`${label} must be a finite number.`);
  }
  return value;
}

function integer(value: unknown, label: string, minimum = 0): number {
  const normalized = numberValue(value, label, minimum);
  if (!Number.isInteger(normalized)) {
    throw new AtlasDataError(`${label} must be an integer.`);
  }
  return normalized;
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new AtlasDataError(`${label} must be a non-empty string.`);
  }
  return value;
}

function exactFields(
  value: Record<string, unknown>,
  fields: readonly string[],
  label: string,
): void {
  const expected = new Set(fields);
  const missing = fields.filter((field) => !(field in value));
  const unsupported = Object.keys(value).filter((field) => !expected.has(field));
  if (missing.length > 0) {
    throw new AtlasDataError(`${label} is missing fields: ${missing.join(", ")}.`);
  }
  if (unsupported.length > 0) {
    throw new AtlasDataError(
      `${label} has unsupported fields: ${unsupported.join(", ")}.`,
    );
  }
}

function vector3(value: unknown, label: string): Vector3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new AtlasDataError(`${label} must contain three numbers.`);
  }
  return [
    numberValue(value[0], label, Number.NEGATIVE_INFINITY),
    numberValue(value[1], label, Number.NEGATIVE_INFINITY),
    numberValue(value[2], label, Number.NEGATIVE_INFINITY),
  ];
}

function schema(value: unknown, label: string): SchemaVersion {
  const candidate = object(value, `${label}.schema`);
  if (candidate.name !== "lerobot-state-atlas.browser-data") {
    throw new AtlasDataError(`${label} uses an unsupported schema name.`);
  }
  if (candidate.major !== 1) {
    throw new AtlasDataError(
      `Unsupported browser-data major version: ${String(candidate.major)}.`,
    );
  }
  return {
    name: candidate.name,
    major: 1,
    minor: integer(candidate.minor, `${label}.schema.minor`),
  };
}

export function decodeManifest(value: unknown): AtlasManifest {
  const candidate = object(value, "Manifest");
  const dataset = object(candidate.dataset, "Manifest dataset");
  const exporter = object(candidate.exporter, "Manifest exporter");
  const robot = object(candidate.robot, "Manifest robot");
  const coverage = object(candidate.coverage, "Manifest coverage");
  const coordinates = object(candidate.coordinates, "Manifest coordinates");
  const totals = object(candidate.totals, "Manifest totals");
  const bounds = object(candidate.sceneBounds, "Manifest scene bounds");
  const canonicalTransforms = object(
    coverage.canonicalTransforms,
    "Canonical transforms",
  );

  const decodedManifestSchema = schema(candidate.schema, "Manifest");
  if (!Array.isArray(candidate.payloads)) {
    throw new AtlasDataError("Manifest payloads must be an array.");
  }
  const payloads = candidate.payloads.map((rawPayload, index) => {
    const payload = object(rawPayload, `Payload ${index}`);
    if (
      payload.kind !== "coverage" &&
      payload.kind !== "trajectories" &&
      payload.kind !== "episode-videos"
    ) {
      throw new AtlasDataError(`Payload ${index} has an unsupported kind.`);
    }
    if (payload.encoding !== "json") {
      throw new AtlasDataError(`Payload ${index} has an unsupported encoding.`);
    }
    return {
      kind: payload.kind,
      filename: stringValue(payload.filename, `Payload ${index} filename`),
      required: Boolean(payload.required),
      encoding: payload.encoding,
      byteSize: integer(payload.byteSize, `Payload ${index} byte size`, 1),
      sha256: stringValue(payload.sha256, `Payload ${index} checksum`),
    };
  });
  if (!payloads.some((payload) => payload.kind === "coverage")) {
    throw new AtlasDataError("Manifest does not reference coverage data.");
  }
  if (
    decodedManifestSchema.minor < 1 &&
    payloads.some((payload) => payload.kind === "episode-videos")
  ) {
    throw new AtlasDataError(
      "Episode-video payloads require browser-data schema v1.1 or newer.",
    );
  }
  if (!Array.isArray(dataset.episodeIds)) {
    throw new AtlasDataError("Dataset episode IDs must be an array.");
  }
  const episodeIds = dataset.episodeIds.map((value, index) =>
    integer(value, `Episode ID ${index}`),
  );
  if (episodeIds.length !== dataset.episodeCount) {
    throw new AtlasDataError("Dataset episode count is inconsistent.");
  }
  stringValue(robot.modelName, "Robot model name");
  stringValue(robot.rootLink, "Robot root link");
  stringValue(robot.urdfSha256, "URDF checksum");
  stringValue(dataset.requestedRevision, "Requested dataset revision");
  const resolvedRevision = stringValue(
    dataset.resolvedRevision,
    "Resolved dataset revision",
  );
  if (!/^[0-9a-f]{40}$/.test(resolvedRevision)) {
    throw new AtlasDataError(
      "Resolved dataset revision must be a full lowercase commit SHA.",
    );
  }
  stringValue(dataset.lerobotCodebaseVersion, "LeRobot codebase version");
  stringValue(dataset.lerobotPackageVersion, "LeRobot package version");
  const repositoryHeadCommit = stringValue(
    exporter.repositoryHeadCommit,
    "Repository HEAD commit",
  );
  if (!/^[0-9a-f]{40}$/.test(repositoryHeadCommit)) {
    throw new AtlasDataError(
      "Repository HEAD commit must be a full lowercase Git commit hash.",
    );
  }
  if (typeof exporter.workingTreeDirty !== "boolean") {
    throw new AtlasDataError("Working-tree dirty state must be boolean.");
  }
  stringValue(exporter.sourceDescription, "Exporter source description");
  stringValue(exporter.determinism, "Exporter determinism description");

  const transform = (arm: "left" | "right") => {
    const value = object(canonicalTransforms[arm], `${arm} transform`);
    return {
      translationXyz: vector3(value.translationXyz, `${arm} translation`),
      rotationRpy: vector3(value.rotationRpy, `${arm} rotation`),
    };
  };
  transform("left");
  transform("right");
  numberValue(coverage.voxelSize, "Voxel size", Number.MIN_VALUE);
  vector3(coverage.voxelOrigin, "Voxel origin");
  vector3(bounds.minimumXyz, "Scene minimum");
  vector3(bounds.maximumXyz, "Scene maximum");
  for (const key of [
    "datasetFrameCount",
    "toolPointVisitCount",
    "armVoxelEntryCount",
    "uniqueSharedGridCellCount",
  ]) {
    integer(totals[key], `Manifest total ${key}`, 1);
  }

  if (
    coordinates.lengthUnit !== "metre" ||
    coordinates.angleUnit !== "radian" ||
    coordinates.handedness !== "right-handed" ||
    coordinates.positionFrame !== "canonical-shared-world"
  ) {
    throw new AtlasDataError("Manifest coordinate conventions are unsupported.");
  }
  if (coverage.spacingCalibrated !== false) {
    throw new AtlasDataError("Demo spacing must be explicitly uncalibrated.");
  }

  return candidate as unknown as AtlasManifest & {
    schema: SchemaVersion;
    dataset: { episodeIds: number[] };
    payloads: typeof payloads;
    coverage: {
      voxelSize: number;
      voxelOrigin: Vector3;
      canonicalTransforms: {
        left: ReturnType<typeof transform>;
        right: ReturnType<typeof transform>;
      };
    };
    sceneBounds: { minimumXyz: Vector3; maximumXyz: Vector3 };
  };
}

function decodeArm(value: unknown, expectedArm: "left" | "right"): CoverageArm {
  const arm = object(value, `${expectedArm} coverage`);
  if (arm.arm !== expectedArm) {
    throw new AtlasDataError(`Expected ${expectedArm} coverage.`);
  }
  const arrayFields = [
    "voxelIndices",
    "visitCounts",
    "episodeCounts",
    "episodeIdOffsets",
    "episodeIds",
  ] as const;
  for (const field of arrayFields) {
    if (!Array.isArray(arm[field])) {
      throw new AtlasDataError(`${expectedArm} ${field} must be an array.`);
    }
  }
  const voxelIndices = (arm.voxelIndices as unknown[]).map((value, index) => {
    const result = vector3(value, `${expectedArm} voxel ${index}`);
    if (!result.every(Number.isInteger)) {
      throw new AtlasDataError("Voxel indices must contain integers.");
    }
    return result;
  });
  const visitCounts = (arm.visitCounts as unknown[]).map((value, index) =>
    integer(value, `${expectedArm} visit ${index}`, 1),
  );
  const episodeCounts = (arm.episodeCounts as unknown[]).map((value, index) =>
    integer(value, `${expectedArm} episode count ${index}`, 1),
  );
  const episodeIdOffsets = (arm.episodeIdOffsets as unknown[]).map(
    (value, index) => integer(value, `${expectedArm} CSR offset ${index}`),
  );
  const episodeIds = (arm.episodeIds as unknown[]).map((value, index) =>
    integer(value, `${expectedArm} episode ID ${index}`),
  );
  if (
    visitCounts.length !== voxelIndices.length ||
    episodeCounts.length !== voxelIndices.length ||
    episodeIdOffsets.length !== voxelIndices.length + 1 ||
    episodeIdOffsets[0] !== 0 ||
    episodeIdOffsets.at(-1) !== episodeIds.length
  ) {
    throw new AtlasDataError(`${expectedArm} coverage arrays are inconsistent.`);
  }
  for (let index = 0; index < voxelIndices.length; index += 1) {
    const count = episodeIdOffsets[index + 1] - episodeIdOffsets[index];
    if (count !== episodeCounts[index]) {
      throw new AtlasDataError(
        `${expectedArm} episode counts disagree with CSR data.`,
      );
    }
  }
  return {
    arm: expectedArm,
    toolLink: stringValue(arm.toolLink, `${expectedArm} tool link`),
    voxelIndices,
    visitCounts,
    episodeCounts,
    episodeIdOffsets,
    episodeIds,
    statistics: object(
      arm.statistics,
      `${expectedArm} statistics`,
    ) as unknown as CoverageArm["statistics"],
  };
}

export function decodeCoverage(value: unknown): CoveragePayload {
  const candidate = object(value, "Coverage payload");
  const decodedSchema = schema(candidate.schema, "Coverage payload");
  if (!Array.isArray(candidate.arms) || candidate.arms.length !== 2) {
    throw new AtlasDataError("Coverage payload must contain two arms.");
  }
  return {
    schema: decodedSchema,
    arms: [
      decodeArm(candidate.arms[0], "left"),
      decodeArm(candidate.arms[1], "right"),
    ],
  };
}

export function decodeTrajectories(value: unknown): TrajectoryPayload {
  const candidate = object(value, "Trajectory payload");
  const decodedSchema = schema(candidate.schema, "Trajectory payload");
  if (!Array.isArray(candidate.episodes) || candidate.episodes.length === 0) {
    throw new AtlasDataError("Trajectory payload must contain episodes.");
  }
  const seen = new Set<number>();
  const episodes = candidate.episodes.map((rawEpisode, episodeIndex) => {
    const episode = object(rawEpisode, `Trajectory episode ${episodeIndex}`);
    const episodeId = integer(
      episode.episodeId,
      `Trajectory episode ${episodeIndex} ID`,
    );
    if (seen.has(episodeId)) {
      throw new AtlasDataError("Trajectory episode IDs must be unique.");
    }
    seen.add(episodeId);
    const arrays = [
      "frameIndices",
      "timestampsSeconds",
      "leftPositionsXyz",
      "rightPositionsXyz",
    ] as const;
    for (const field of arrays) {
      if (!Array.isArray(episode[field])) {
        throw new AtlasDataError(`Trajectory episode ${episodeId} ${field} must be an array.`);
      }
    }
    const frameIndices = (episode.frameIndices as unknown[]).map((item, index) =>
      integer(item, `Trajectory episode ${episodeId} frame ${index}`),
    );
    const timestampsSeconds = (episode.timestampsSeconds as unknown[]).map(
      (item, index) =>
        numberValue(item, `Trajectory episode ${episodeId} timestamp ${index}`),
    );
    const leftPositionsXyz = (episode.leftPositionsXyz as unknown[]).map(
      (item, index) => vector3(item, `Trajectory episode ${episodeId} left position ${index}`),
    );
    const rightPositionsXyz = (episode.rightPositionsXyz as unknown[]).map(
      (item, index) => vector3(item, `Trajectory episode ${episodeId} right position ${index}`),
    );
    const length = frameIndices.length;
    if (
      length === 0 ||
      timestampsSeconds.length !== length ||
      leftPositionsXyz.length !== length ||
      rightPositionsXyz.length !== length
    ) {
      throw new AtlasDataError(`Trajectory episode ${episodeId} arrays are inconsistent.`);
    }
    for (let index = 1; index < length; index += 1) {
      if (
        frameIndices[index] <= frameIndices[index - 1] ||
        timestampsSeconds[index] < timestampsSeconds[index - 1]
      ) {
        throw new AtlasDataError(
          `Trajectory episode ${episodeId} frames and timestamps must be ordered.`,
        );
      }
    }
    return {
      episodeId,
      frameIndices,
      timestampsSeconds,
      leftPositionsXyz,
      rightPositionsXyz,
    } satisfies TrajectoryEpisode;
  });
  return { schema: decodedSchema, episodes };
}

function episodeVideoFilename(value: unknown, label: string): string {
  const filename = stringValue(value, label);
  const parts = filename.split("/");
  if (
    filename.includes("\\") ||
    /[:?#%]/.test(filename) ||
    filename.startsWith("/") ||
    parts.some((part) => part === "" || part === "." || part === "..") ||
    !filename.toLowerCase().endsWith(".mp4")
  ) {
    throw new AtlasDataError(
      `${label} must be a safe bundle-relative MP4 path.`,
    );
  }
  return filename;
}

export function decodeEpisodeVideos(value: unknown): EpisodeVideoPayload {
  const candidate = object(value, "Episode-video payload");
  exactFields(
    candidate,
    ["schema", "defaultCameraId", "cameras", "episodes"],
    "Episode-video payload",
  );
  const decodedSchema = schema(candidate.schema, "Episode-video payload");
  if (decodedSchema.minor < 1) {
    throw new AtlasDataError(
      "Episode-video payload requires browser-data schema v1.1 or newer.",
    );
  }

  if (!Array.isArray(candidate.cameras) || candidate.cameras.length === 0) {
    throw new AtlasDataError("Episode-video payload must contain cameras.");
  }

  const cameraIds = new Set<string>();
  let previousCameraId: string | undefined;
  const cameras = candidate.cameras.map((rawCamera, index) => {
    const camera = object(rawCamera, `Episode-video camera ${index}`);
    exactFields(
      camera,
      ["cameraId", "datasetFeature", "label", "width", "height"],
      `Episode-video camera ${index}`,
    );
    const cameraId = stringValue(
      camera.cameraId,
      `Episode-video camera ${index} ID`,
    );
    if (cameraIds.has(cameraId)) {
      throw new AtlasDataError("Episode-video camera IDs must be unique.");
    }
    if (previousCameraId !== undefined && cameraId < previousCameraId) {
      throw new AtlasDataError(
        "Episode-video cameras must be ordered by camera ID.",
      );
    }
    cameraIds.add(cameraId);
    previousCameraId = cameraId;
    return {
      cameraId,
      datasetFeature: stringValue(
        camera.datasetFeature,
        `Episode-video camera ${cameraId} dataset feature`,
      ),
      label: stringValue(
        camera.label,
        `Episode-video camera ${cameraId} label`,
      ),
      width: integer(
        camera.width,
        `Episode-video camera ${cameraId} width`,
        1,
      ),
      height: integer(
        camera.height,
        `Episode-video camera ${cameraId} height`,
        1,
      ),
    } satisfies EpisodeVideoCamera;
  });

  const defaultCameraId = stringValue(
    candidate.defaultCameraId,
    "Episode-video default camera ID",
  );
  if (!cameraIds.has(defaultCameraId)) {
    throw new AtlasDataError(
      "Episode-video default camera must identify a declared camera.",
    );
  }

  if (!Array.isArray(candidate.episodes) || candidate.episodes.length === 0) {
    throw new AtlasDataError("Episode-video payload must contain episodes.");
  }

  const episodeIds = new Set<number>();
  const filenames = new Set<string>();
  let previousEpisodeId: number | undefined;
  const episodes = candidate.episodes.map((rawEpisode, episodeIndex) => {
    const episode = object(
      rawEpisode,
      `Episode-video episode ${episodeIndex}`,
    );
    exactFields(
      episode,
      ["episodeId", "videos"],
      `Episode-video episode ${episodeIndex}`,
    );
    const episodeId = integer(
      episode.episodeId,
      `Episode-video episode ${episodeIndex} ID`,
    );
    if (episodeIds.has(episodeId)) {
      throw new AtlasDataError("Episode-video episode IDs must be unique.");
    }
    if (previousEpisodeId !== undefined && episodeId < previousEpisodeId) {
      throw new AtlasDataError(
        "Episode-video episodes must be ordered by episode ID.",
      );
    }
    episodeIds.add(episodeId);
    previousEpisodeId = episodeId;

    if (!Array.isArray(episode.videos) || episode.videos.length === 0) {
      throw new AtlasDataError(
        `Episode-video episode ${episodeId} must contain videos.`,
      );
    }

    const sourceCameraIds = new Set<string>();
    let previousSourceCameraId: string | undefined;
    const videos = episode.videos.map((rawSource, sourceIndex) => {
      const source = object(
        rawSource,
        `Episode-video episode ${episodeId} source ${sourceIndex}`,
      );
      exactFields(
        source,
        [
          "cameraId",
          "filename",
          "mimeType",
          "fromTimestampSeconds",
          "toTimestampSeconds",
          "byteSize",
          "sha256",
        ],
        `Episode-video episode ${episodeId} source ${sourceIndex}`,
      );
      const cameraId = stringValue(
        source.cameraId,
        `Episode-video episode ${episodeId} source camera ID`,
      );
      if (!cameraIds.has(cameraId)) {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} references an undeclared camera.`,
        );
      }
      if (sourceCameraIds.has(cameraId)) {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} camera IDs must be unique.`,
        );
      }
      if (
        previousSourceCameraId !== undefined &&
        cameraId < previousSourceCameraId
      ) {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} videos must be ordered by camera ID.`,
        );
      }
      sourceCameraIds.add(cameraId);
      previousSourceCameraId = cameraId;

      const filename = episodeVideoFilename(
        source.filename,
        `Episode-video episode ${episodeId} filename`,
      );
      if (filenames.has(filename)) {
        throw new AtlasDataError(
          "Episode-video media filenames must be globally unique.",
        );
      }
      filenames.add(filename);

      if (source.mimeType !== "video/mp4") {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} MIME type must be video/mp4.`,
        );
      }

      const fromTimestampSeconds = numberValue(
        source.fromTimestampSeconds,
        `Episode-video episode ${episodeId} start timestamp`,
      );
      const toTimestampSeconds = numberValue(
        source.toTimestampSeconds,
        `Episode-video episode ${episodeId} end timestamp`,
      );
      if (toTimestampSeconds <= fromTimestampSeconds) {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} timestamps are invalid.`,
        );
      }

      const sha256 = stringValue(
        source.sha256,
        `Episode-video episode ${episodeId} checksum`,
      );
      if (!/^[0-9a-f]{64}$/.test(sha256)) {
        throw new AtlasDataError(
          `Episode-video episode ${episodeId} checksum must be lowercase SHA-256.`,
        );
      }

      return {
        cameraId,
        filename,
        mimeType: source.mimeType,
        fromTimestampSeconds,
        toTimestampSeconds,
        byteSize: integer(
          source.byteSize,
          `Episode-video episode ${episodeId} byte size`,
          1,
        ),
        sha256,
      } satisfies EpisodeVideoSource;
    });

    if (!sourceCameraIds.has(defaultCameraId)) {
      throw new AtlasDataError(
        `Episode-video episode ${episodeId} must include the default camera.`,
      );
    }

    return { episodeId, videos } satisfies EpisodeVideoEpisode;
  });

  return {
    schema: decodedSchema,
    defaultCameraId,
    cameras,
    episodes,
  };
}
