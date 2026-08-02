import type {
  EnvironmentAlignment,
  EnvironmentAssetReference,
  EnvironmentBounds,
  EnvironmentManifest,
  EnvironmentProvenance,
  EnvironmentQuaternionXyzw,
  EnvironmentSchemaVersion,
  EnvironmentVector3,
} from "./types";

export class EnvironmentManifestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EnvironmentManifestError";
  }
}

function fail(message: string): never {
  throw new EnvironmentManifestError(message);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${path} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function exactFields(
  value: Record<string, unknown>,
  expectedFields: readonly string[],
  path: string,
): void {
  const expected = new Set(expectedFields);
  const missing = expectedFields.filter((field) => !(field in value));
  const unsupported = Object.keys(value).filter((field) => !expected.has(field));
  if (missing.length > 0) fail(`${path} is missing fields: ${missing.join(", ")}.`);
  if (unsupported.length > 0) {
    fail(`${path} has unsupported fields: ${unsupported.join(", ")}.`);
  }
}

function nonEmptyString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(`${path} must be a non-empty string.`);
  }
  return value;
}

function finiteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${path} must be a finite number.`);
  }
  return value;
}

function positiveInteger(value: unknown, path: string): number {
  const normalized = finiteNumber(value, path);
  if (!Number.isInteger(normalized) || normalized < 1) {
    fail(`${path} must be a positive integer.`);
  }
  return normalized;
}

function vector3(value: unknown, path: string): EnvironmentVector3 {
  if (!Array.isArray(value) || value.length !== 3) {
    fail(`${path} must contain exactly three finite numbers.`);
  }
  return value.map((item, index) =>
    finiteNumber(item, `${path}[${index}]`),
  ) as EnvironmentVector3;
}

function quaternion(
  value: unknown,
  path: string,
): EnvironmentQuaternionXyzw {
  if (!Array.isArray(value) || value.length !== 4) {
    fail(`${path} must contain exactly four finite XYZW components.`);
  }
  const normalized = value.map((item, index) =>
    finiteNumber(item, `${path}[${index}]`),
  ) as EnvironmentQuaternionXyzw;
  const norm = Math.hypot(...normalized);
  if (Math.abs(norm - 1) > 1e-6) fail(`${path} must be a unit quaternion.`);
  return normalized;
}

function schema(value: unknown): EnvironmentSchemaVersion {
  const candidate = record(value, "manifest.schema");
  exactFields(candidate, ["name", "major", "minor"], "manifest.schema");
  if (
    candidate.name !== "lerobot-state-atlas.environment-layer" ||
    candidate.major !== 1 ||
    candidate.minor !== 0
  ) {
    fail("manifest.schema must be lerobot-state-atlas.environment-layer v1.0.");
  }
  return {
    name: "lerobot-state-atlas.environment-layer",
    major: 1,
    minor: 0,
  };
}

function provenance(value: unknown): EnvironmentProvenance {
  const candidate = record(value, "manifest.provenance");
  exactFields(
    candidate,
    ["sourceKind", "description", "reconstructionClaim"],
    "manifest.provenance",
  );
  if (candidate.sourceKind !== "real-scan" && candidate.sourceKind !== "synthetic-test") {
    fail("manifest.provenance.sourceKind is unsupported.");
  }
  const description = nonEmptyString(
    candidate.description,
    "manifest.provenance.description",
  );
  if (
    candidate.reconstructionClaim !== false &&
    candidate.reconstructionClaim !== "documented-real-scan"
  ) {
    fail("manifest.provenance.reconstructionClaim is unsupported.");
  }
  if (
    candidate.sourceKind === "synthetic-test" &&
    candidate.reconstructionClaim !== false
  ) {
    fail("Synthetic environments must not claim a real reconstruction.");
  }
  if (
    candidate.sourceKind === "synthetic-test" &&
    !description.toLowerCase().includes("synthetic")
  ) {
    fail("Synthetic environment descriptions must explicitly say synthetic.");
  }
  return {
    sourceKind: candidate.sourceKind,
    description,
    reconstructionClaim: candidate.reconstructionClaim,
  };
}

function alignment(value: unknown): EnvironmentAlignment {
  const candidate = record(value, "manifest.alignment");
  exactFields(
    candidate,
    ["translationXyz", "rotationXyzw", "uniformScale", "calibrated", "disclosure"],
    "manifest.alignment",
  );
  const uniformScale = finiteNumber(candidate.uniformScale, "manifest.alignment.uniformScale");
  if (uniformScale <= 0) fail("manifest.alignment.uniformScale must be greater than zero.");
  if (typeof candidate.calibrated !== "boolean") {
    fail("manifest.alignment.calibrated must be boolean.");
  }
  return {
    translationXyz: vector3(candidate.translationXyz, "manifest.alignment.translationXyz"),
    rotationXyzw: quaternion(candidate.rotationXyzw, "manifest.alignment.rotationXyzw"),
    uniformScale,
    calibrated: candidate.calibrated,
    disclosure: nonEmptyString(candidate.disclosure, "manifest.alignment.disclosure"),
  };
}

function bounds(value: unknown): EnvironmentBounds {
  const candidate = record(value, "manifest.bounds");
  exactFields(candidate, ["minimumXyz", "maximumXyz"], "manifest.bounds");
  const minimumXyz = vector3(candidate.minimumXyz, "manifest.bounds.minimumXyz");
  const maximumXyz = vector3(candidate.maximumXyz, "manifest.bounds.maximumXyz");
  if (minimumXyz.some((minimum, axis) => minimum >= maximumXyz[axis])) {
    fail("manifest.bounds minimum values must be less than maximum values.");
  }
  return { minimumXyz, maximumXyz };
}

function asset(value: unknown): EnvironmentAssetReference {
  const candidate = record(value, "manifest.asset");
  exactFields(
    candidate,
    ["filename", "format", "mimeType", "byteSize", "sha256", "splatCount"],
    "manifest.asset",
  );
  const filename = nonEmptyString(candidate.filename, "manifest.asset.filename");
  if (
    filename !== filename.trim() ||
    filename === "." ||
    filename === ".." ||
    filename.includes("/") ||
    filename.includes("\\") ||
    filename.includes("?") ||
    filename.includes("#") ||
    /%2f|%5c/i.test(filename) ||
    !/^[A-Za-z0-9._-]+\.spz$/.test(filename)
  ) {
    fail("manifest.asset.filename must be a safe single-segment .spz filename.");
  }
  if (candidate.format !== "spz") fail("manifest.asset.format must be spz.");
  if (candidate.mimeType !== "application/octet-stream") {
    fail("manifest.asset.mimeType must be application/octet-stream.");
  }
  if (typeof candidate.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(candidate.sha256)) {
    fail("manifest.asset.sha256 must be a lowercase SHA-256 digest.");
  }
  return {
    filename,
    format: "spz",
    mimeType: "application/octet-stream",
    byteSize: positiveInteger(candidate.byteSize, "manifest.asset.byteSize"),
    sha256: candidate.sha256,
    splatCount: positiveInteger(candidate.splatCount, "manifest.asset.splatCount"),
  };
}

export function decodeEnvironmentManifest(value: unknown): EnvironmentManifest {
  const candidate = record(value, "manifest");
  const commonFields = [
    "schema",
    "environmentId",
    "label",
    "status",
    "provenance",
    "coordinateFrame",
  ] as const;
  if (candidate.status === "unavailable") {
    exactFields(candidate, [...commonFields, "unavailableReason"], "manifest");
  } else if (candidate.status === "available") {
    exactFields(candidate, [...commonFields, "alignment", "bounds", "asset"], "manifest");
  } else {
    fail("manifest.status must be available or unavailable.");
  }

  const normalizedSchema = schema(candidate.schema);
  const environmentId = nonEmptyString(candidate.environmentId, "manifest.environmentId");
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(environmentId)) {
    fail("manifest.environmentId must be a safe lowercase identifier.");
  }
  const label = nonEmptyString(candidate.label, "manifest.label");
  const normalizedProvenance = provenance(candidate.provenance);
  if (candidate.coordinateFrame !== "canonical-shared-world") {
    fail("manifest.coordinateFrame must be canonical-shared-world.");
  }

  const common = {
    schema: normalizedSchema,
    environmentId,
    label,
    provenance: normalizedProvenance,
    coordinateFrame: "canonical-shared-world" as const,
  };
  if (candidate.status === "unavailable") {
    return {
      ...common,
      status: "unavailable",
      unavailableReason: nonEmptyString(
        candidate.unavailableReason,
        "manifest.unavailableReason",
      ),
    };
  }
  return {
    ...common,
    status: "available",
    alignment: alignment(candidate.alignment),
    bounds: bounds(candidate.bounds),
    asset: asset(candidate.asset),
  };
}
