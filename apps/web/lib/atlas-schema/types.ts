export type Vector3 = [number, number, number];
export type QuaternionXyzw = [number, number, number, number];

export interface SchemaVersion {
  name: "lerobot-state-atlas.browser-data";
  major: number;
  minor: number;
}

export interface RigidTransform {
  translationXyz: Vector3;
  rotationRpy: Vector3;
}

export interface PayloadReference {
  kind: "coverage" | "trajectories" | "episode-videos";
  filename: string;
  required: boolean;
  encoding: "json";
  byteSize: number;
  sha256: string;
}

export interface AtlasManifest {
  schema: SchemaVersion;
  bundleId: string;
  exporter: {
    packageVersion: string;
    repositoryHeadCommit: string;
    workingTreeDirty: boolean;
    sourceDescription: string;
    determinism: string;
  };
  dataset: {
    repositoryId: string;
    requestedRevision: string;
    resolvedRevision: string;
    lerobotCodebaseVersion: string;
    lerobotPackageVersion: string;
    robotType: string | null;
    fps: number;
    episodeIds: number[];
    episodeCount: number;
    datasetFrameCount: number;
  };
  robot: {
    modelName: string;
    rootLink: string;
    toolLinks: Record<"left" | "right", string>;
    urdfSha256: string;
    urdfUpstreamIdentity: string;
  };
  coverage: {
    voxelSize: number;
    voxelOrigin: Vector3;
    canonicalTransforms: Record<"left" | "right", RigidTransform>;
    armSpacing: number;
    spacingCalibrated: false;
    spacingDisclosure: string;
  };
  coordinates: {
    lengthUnit: "metre";
    angleUnit: "radian";
    handedness: "right-handed";
    positionFrame: "canonical-shared-world";
    voxelIndexFormula: string;
    rotationConvention: string;
    pointTransform: string;
  };
  totals: {
    datasetFrameCount: number;
    toolPointVisitCount: number;
    armVoxelEntryCount: number;
    uniqueSharedGridCellCount: number;
  };
  sceneBounds: {
    minimumXyz: Vector3;
    maximumXyz: Vector3;
  };
  payloads: PayloadReference[];
  trajectoryState?: TrajectoryStateManifest;
}

export interface TrajectoryStateManifest {
  orientation: {
    available: boolean;
    representation: "unit-quaternion";
    componentOrder: ["x", "y", "z", "w"];
    frame: "canonical-shared-world";
    samplePolicy: "recorded-sample";
  };
  gripper: {
    available: boolean;
    leftSourceComponent: "left_gripper.pos";
    rightSourceComponent: "right_gripper.pos";
    valueSemantics: "raw-device-specific-unproven";
    physicalJawWidthCalibrated: false;
    polarityEstablished: false;
    visualizationGeometryCalibrated: false;
  };
}

export type OptionalTrajectoryCapability<T> =
  | { status: "available"; data: T }
  | { status: "unavailable" }
  | { status: "degraded"; warning: string };

export interface TrajectoryEpisodeOrientations {
  episodeId: number;
  leftOrientationsXyzw: QuaternionXyzw[];
  rightOrientationsXyzw: QuaternionXyzw[];
}

export interface TrajectoryOrientationData {
  episodes: TrajectoryEpisodeOrientations[];
}

export interface TrajectoryEpisodeRecordedGripperValues {
  episodeId: number;
  leftRecordedGripperValues: number[];
  rightRecordedGripperValues: number[];
}

export interface TrajectoryRecordedGripperData {
  episodes: TrajectoryEpisodeRecordedGripperValues[];
}

export interface CoverageArm {
  arm: "left" | "right";
  toolLink: string;
  voxelIndices: Vector3[];
  visitCounts: number[];
  episodeCounts: number[];
  episodeIdOffsets: number[];
  episodeIds: number[];
  statistics: {
    voxelEntryCount: number;
    minimumVisitCount: number;
    maximumVisitCount: number;
    minimumEpisodeCount: number;
    maximumEpisodeCount: number;
  };
}

export interface CoveragePayload {
  schema: SchemaVersion;
  arms: CoverageArm[];
}

export interface TrajectoryEpisode {
  episodeId: number;
  frameIndices: number[];
  timestampsSeconds: number[];
  leftPositionsXyz: Vector3[];
  rightPositionsXyz: Vector3[];
}

export interface TrajectoryPayload {
  schema: SchemaVersion;
  episodes: TrajectoryEpisode[];
  orientation: OptionalTrajectoryCapability<TrajectoryOrientationData>;
  gripper: OptionalTrajectoryCapability<TrajectoryRecordedGripperData>;
}

export interface EpisodeVideoCamera {
  cameraId: string;
  datasetFeature: string;
  label: string;
  width: number;
  height: number;
}

export interface EpisodeVideoSource {
  cameraId: string;
  filename: string;
  mimeType: "video/mp4";
  fromTimestampSeconds: number;
  toTimestampSeconds: number;
  byteSize: number;
  sha256: string;
}

export interface EpisodeVideoEpisode {
  episodeId: number;
  videos: EpisodeVideoSource[];
}

export interface EpisodeVideoPayload {
  schema: SchemaVersion;
  defaultCameraId: string;
  cameras: EpisodeVideoCamera[];
  episodes: EpisodeVideoEpisode[];
}

export interface PreparedVoxelArm {
  arm: "left" | "right";
  centers: Float32Array;
  visits: Uint32Array;
  episodeCounts: Uint32Array;
  instanceLookup: readonly {
    arm: "left" | "right";
    voxelEntryIndex: number;
    voxelIndex: Vector3;
  }[];
  minimumVisitCount: number;
  maximumVisitCount: number;
}

export interface AtlasData {
  manifest: AtlasManifest;
  coverage: CoveragePayload;
  preparedArms: PreparedVoxelArm[];
}
