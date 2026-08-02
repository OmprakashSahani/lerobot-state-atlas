export type EnvironmentVector3 = [number, number, number];
export type EnvironmentQuaternionXyzw = [number, number, number, number];

export interface EnvironmentSchemaVersion {
  name: "lerobot-state-atlas.environment-layer";
  major: 1;
  minor: 0;
}

export interface EnvironmentProvenance {
  sourceKind: "real-scan" | "synthetic-test";
  description: string;
  reconstructionClaim: "documented-real-scan" | false;
}

export interface EnvironmentAlignment {
  translationXyz: EnvironmentVector3;
  rotationXyzw: EnvironmentQuaternionXyzw;
  uniformScale: number;
  calibrated: boolean;
  disclosure: string;
}

export interface EnvironmentBounds {
  minimumXyz: EnvironmentVector3;
  maximumXyz: EnvironmentVector3;
}

export interface EnvironmentAssetReference {
  filename: string;
  format: "spz";
  mimeType: "application/octet-stream";
  byteSize: number;
  sha256: string;
  splatCount: number;
}

interface EnvironmentManifestBase {
  schema: EnvironmentSchemaVersion;
  environmentId: string;
  label: string;
  provenance: EnvironmentProvenance;
  coordinateFrame: "canonical-shared-world";
}

export interface AvailableEnvironmentManifest extends EnvironmentManifestBase {
  status: "available";
  alignment: EnvironmentAlignment;
  bounds: EnvironmentBounds;
  asset: EnvironmentAssetReference;
}

export interface UnavailableEnvironmentManifest extends EnvironmentManifestBase {
  status: "unavailable";
  unavailableReason: string;
}

export type EnvironmentManifest =
  | AvailableEnvironmentManifest
  | UnavailableEnvironmentManifest;

export type EnvironmentCapability =
  | { status: "unavailable"; reason: string }
  | { status: "available"; manifest: AvailableEnvironmentManifest };

export const demoEnvironmentCapability: EnvironmentCapability = {
  status: "unavailable",
  reason:
    "No validated Gaussian Splat scan or environment-to-robot calibration is bundled with this demo.",
};
