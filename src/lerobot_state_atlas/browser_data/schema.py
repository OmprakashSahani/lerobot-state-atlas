"""Constants and structural rules for browser-data schema version 1.1."""

SCHEMA_NAME = "lerobot-state-atlas.browser-data"
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 1

MANIFEST_FILENAME = "manifest.json"
COVERAGE_FILENAME = "coverage.json"
TRAJECTORY_FILENAME = "trajectories.json"
EPISODE_VIDEO_FILENAME = "episode-videos.json"

MANIFEST_FIELDS = {
    "schema",
    "bundleId",
    "exporter",
    "dataset",
    "robot",
    "coverage",
    "coordinates",
    "totals",
    "sceneBounds",
    "payloads",
}

SCHEMA_FIELDS = {"name", "major", "minor"}
EXPORTER_FIELDS = {
    "packageVersion",
    "repositoryHeadCommit",
    "workingTreeDirty",
    "sourceDescription",
    "determinism",
}
DATASET_FIELDS = {
    "repositoryId",
    "requestedRevision",
    "resolvedRevision",
    "lerobotCodebaseVersion",
    "lerobotPackageVersion",
    "robotType",
    "fps",
    "episodeIds",
    "episodeCount",
    "datasetFrameCount",
}
ROBOT_FIELDS = {
    "modelName",
    "rootLink",
    "toolLinks",
    "urdfSha256",
    "urdfUpstreamIdentity",
}
COVERAGE_CONFIG_FIELDS = {
    "voxelSize",
    "voxelOrigin",
    "canonicalTransforms",
    "armSpacing",
    "spacingCalibrated",
    "spacingDisclosure",
}
TRANSFORM_FIELDS = {"translationXyz", "rotationRpy"}
COORDINATE_FIELDS = {
    "lengthUnit",
    "angleUnit",
    "handedness",
    "positionFrame",
    "voxelIndexFormula",
    "rotationConvention",
    "pointTransform",
}
TOTAL_FIELDS = {
    "datasetFrameCount",
    "toolPointVisitCount",
    "armVoxelEntryCount",
    "uniqueSharedGridCellCount",
}
BOUNDS_FIELDS = {"minimumXyz", "maximumXyz"}
PAYLOAD_REFERENCE_FIELDS = {
    "kind",
    "filename",
    "required",
    "encoding",
    "byteSize",
    "sha256",
}

COVERAGE_PAYLOAD_FIELDS = {"schema", "arms"}
COVERAGE_ARM_FIELDS = {
    "arm",
    "toolLink",
    "voxelIndices",
    "visitCounts",
    "episodeCounts",
    "episodeIdOffsets",
    "episodeIds",
    "statistics",
}
COVERAGE_STATISTIC_FIELDS = {
    "voxelEntryCount",
    "minimumVisitCount",
    "maximumVisitCount",
    "minimumEpisodeCount",
    "maximumEpisodeCount",
}

TRAJECTORY_PAYLOAD_FIELDS = {"schema", "episodes"}
TRAJECTORY_EPISODE_FIELDS = {
    "episodeId",
    "frameIndices",
    "timestampsSeconds",
    "leftPositionsXyz",
    "rightPositionsXyz",
}

EPISODE_VIDEO_PAYLOAD_FIELDS = {
    "schema",
    "defaultCameraId",
    "cameras",
    "episodes",
}
EPISODE_VIDEO_CAMERA_FIELDS = {
    "cameraId",
    "datasetFeature",
    "label",
    "width",
    "height",
}
EPISODE_VIDEO_EPISODE_FIELDS = {"episodeId", "videos"}
EPISODE_VIDEO_SOURCE_FIELDS = {
    "cameraId",
    "filename",
    "mimeType",
    "fromTimestampSeconds",
    "toTimestampSeconds",
    "byteSize",
    "sha256",
}
