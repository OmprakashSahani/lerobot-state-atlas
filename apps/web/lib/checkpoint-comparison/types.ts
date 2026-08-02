export type Vector3 = [number, number, number];
export type QuaternionXyzw = [number, number, number, number];

export interface ComparisonManifest {
  schema: { name: "lerobot-state-atlas.checkpoint-comparison"; major: 1; minor: 1 };
  bundleId: string;
  dataset: { repositoryId: string; revision: string };
  observation: { observationId: string };
  policies: [
    { policyId: "base-pi05"; label: "Base π0.5"; repositoryId: string; revision: string },
    { policyId: "fine-tuned-pi05"; label: "Fine-tuned π0.5"; repositoryId: string; revision: string },
  ];
  payloads: [{ kind: "plans"; filename: string; encoding: "json"; byteSize: number; sha256: string }];
}

export interface GeneratedPlan {
  policyId: "base-pi05" | "fine-tuned-pi05";
  relativeTimesSeconds: number[];
  actions: number[][];
}

export interface ProjectedArm {
  armId: "left" | "right";
  targetLink: string;
  positionsXyz: Vector3[];
  orientationsXyzw: QuaternionXyzw[];
  generatedRawGripperTargets: number[];
}

export interface ProjectedPlan {
  policyId: "base-pi05" | "fine-tuned-pi05";
  relativeTimesSeconds: number[];
  left: ProjectedArm;
  right: ProjectedArm;
  jointLimitViolations: Array<{
    policyId: string;
    stepIndex: number;
    componentName: string;
    urdfJointName: string;
    value: number;
    bound: number;
    violationKind: "lower" | "upper";
  }>;
}

export interface AvailableProjection {
  available: true;
  sharedConfiguration: true;
  actionInterpretation: {
    interpretationId: string;
    interpretationVersion: string;
    useRelativeActions: false;
    deltaActionsPreprocessorEnabled: false;
    absoluteActionsPostprocessorEnabled: false;
    initialStateParticipatesInTransformation: false;
    componentNames: string[];
    transformationsPerformed: string[];
  };
  initialState: { featureName: string; componentNames: string[]; values: number[]; sha256: string };
  robot: {
    robotModelName: string;
    rootLink: string;
    targetLink: string;
    urdfSha256: string;
    fkImplementationId: string;
    fkImplementationVersion: string;
    outputCoordinateFrame: string;
    lengthUnit: string;
    angularUnit: string;
    handedness: string;
    rotationRepresentation: "quaternion";
    rotationComponentOrder: ["X", "Y", "Z", "W"];
    calibratedArmTransforms: boolean;
    calibratedGripperGeometry: false;
    generatedGripperSemantics: string;
    gripperSemanticDisclaimer: string;
    leftArmTransform: { translationXyz: Vector3; rotationRpy: Vector3 };
    rightArmTransform: { translationXyz: Vector3; rotationRpy: Vector3 };
  };
  jointLimitPolicy: "reject" | "allow-with-recorded-violations";
  plans: [ProjectedPlan, ProjectedPlan];
}

export interface UnavailableProjection {
  available: false;
  reason: string;
}

export interface ComparisonPlans {
  schema: ComparisonManifest["schema"];
  observationId: string;
  actionDimension: 14;
  chunkLength: 50;
  plans: [GeneratedPlan, GeneratedPlan];
  trajectoryProjection: AvailableProjection | UnavailableProjection;
}

export interface CheckpointComparisonData {
  manifest: ComparisonManifest;
  plans: ComparisonPlans;
}
