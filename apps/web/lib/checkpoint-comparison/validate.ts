import type {
  AvailableProjection,
  CheckpointComparisonData,
  ComparisonManifest,
  ComparisonPlans,
  ProjectedArm,
} from "./types";

export class CheckpointComparisonDataError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CheckpointComparisonDataError";
  }
}

const POLICY_IDS = ["base-pi05", "fine-tuned-pi05"] as const;
const POLICY_LABELS = ["Base π0.5", "Fine-tuned π0.5"] as const;
const DATASET_REPOSITORY_ID = "DreamMachines/actuator_unboxing_4h_diverse";
const DATASET_REVISION = "e973df866c80f52884cc68355579043cab828e78";
const PLAN_FPS = 50;
const TIMESTAMP_TOLERANCE = Math.max(1e-9, (1 / PLAN_FPS) * 1e-6);
const ARM_COMPONENTS = [
  "left_joint_1.pos",
  "left_joint_2.pos",
  "left_joint_3.pos",
  "left_joint_4.pos",
  "left_joint_5.pos",
  "left_joint_6.pos",
  "right_joint_1.pos",
  "right_joint_2.pos",
  "right_joint_3.pos",
  "right_joint_4.pos",
  "right_joint_5.pos",
  "right_joint_6.pos",
] as const;

function record(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new CheckpointComparisonDataError(`${path} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new CheckpointComparisonDataError(`${path} must be finite.`);
  }
  return value;
}

function exactFields(value: Record<string, unknown>, fields: readonly string[], path: string): void {
  const expected = new Set(fields);
  const missing = fields.filter((field) => !(field in value));
  const unexpected = Object.keys(value).filter((field) => !expected.has(field));
  if (missing.length > 0) {
    throw new CheckpointComparisonDataError(`${path} is missing fields: ${missing.join(", ")}.`);
  }
  if (unexpected.length > 0) {
    throw new CheckpointComparisonDataError(`${path} has unexpected fields: ${unexpected.join(", ")}.`);
  }
}

function integer(value: unknown, path: string, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw new CheckpointComparisonDataError(`${path} must be an integer from ${minimum} through ${maximum}.`);
  }
  return value;
}

function exactArray(value: unknown, length: number, path: string): unknown[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new CheckpointComparisonDataError(`${path} must contain exactly ${length} items.`);
  }
  return value;
}

function schema(value: unknown, path: string) {
  const item = record(value, path);
  if (
    item.name !== "lerobot-state-atlas.checkpoint-comparison" ||
    item.major !== 1 ||
    item.minor !== 1
  ) {
    throw new CheckpointComparisonDataError(`${path} must be checkpoint-comparison schema 1.1.`);
  }
}

function samples(value: unknown, width: number, path: string): number[][] {
  return exactArray(value, 50, path).map((row, index) =>
    exactArray(row, width, `${path}[${index}]`).map((entry, component) =>
      finite(entry, `${path}[${index}][${component}]`),
    ),
  );
}

function relativeTimes(value: unknown, path: string): number[] {
  const times = exactArray(value, 50, path).map((entry, index) =>
    finite(entry, `${path}[${index}]`),
  );
  if (times[0] !== 0) {
    throw new CheckpointComparisonDataError(`${path}[0] must equal 0.`);
  }
  for (let index = 1; index < times.length; index += 1) {
    if (times[index] <= times[index - 1]) {
      throw new CheckpointComparisonDataError(
        `${path}[${index}] must be greater than the previous timestamp.`,
      );
    }
    const expected = index / PLAN_FPS;
    if (Math.abs(times[index] - expected) > TIMESTAMP_TOLERANCE) {
      throw new CheckpointComparisonDataError(
        `${path}[${index}] must equal ${index} / ${PLAN_FPS} within tolerance ${TIMESTAMP_TOLERANCE}.`,
      );
    }
  }
  return times;
}

function arm(value: unknown, expectedArm: "left" | "right", targetLink: string, path: string): ProjectedArm {
  const item = record(value, path);
  if (item.armId !== expectedArm) throw new CheckpointComparisonDataError(`${path}.armId must equal ${expectedArm}.`);
  if (item.targetLink !== targetLink) throw new CheckpointComparisonDataError(`${path}.targetLink must match robot.targetLink.`);
  const positions = samples(item.positionsXyz, 3, `${path}.positionsXyz`);
  const orientations = samples(item.orientationsXyzw, 4, `${path}.orientationsXyzw`);
  orientations.forEach((quaternion, index) => {
    const norm = Math.sqrt(quaternion.reduce((sum, component) => sum + component * component, 0));
    if (Math.abs(norm - 1) > 1e-6) throw new CheckpointComparisonDataError(`${path}.orientationsXyzw[${index}] must be a unit quaternion.`);
  });
  const gripper = exactArray(item.generatedRawGripperTargets, 50, `${path}.generatedRawGripperTargets`).map((entry, index) => finite(entry, `${path}.generatedRawGripperTargets[${index}]`));
  return { armId: expectedArm, targetLink, positionsXyz: positions as ProjectedArm["positionsXyz"], orientationsXyzw: orientations as ProjectedArm["orientationsXyzw"], generatedRawGripperTargets: gripper };
}

export function decodeComparisonManifest(value: unknown): ComparisonManifest {
  const manifest = record(value, "manifest");
  schema(manifest.schema, "manifest.schema");
  if (typeof manifest.bundleId !== "string" || !manifest.bundleId) throw new CheckpointComparisonDataError("manifest.bundleId must be non-empty.");
  const dataset = record(manifest.dataset, "manifest.dataset");
  exactFields(dataset, ["repositoryId", "revision"], "manifest.dataset");
  if (dataset.repositoryId !== DATASET_REPOSITORY_ID) {
    throw new CheckpointComparisonDataError(`manifest.dataset.repositoryId must equal ${DATASET_REPOSITORY_ID}.`);
  }
  if (dataset.revision !== DATASET_REVISION) {
    throw new CheckpointComparisonDataError(`manifest.dataset.revision must equal ${DATASET_REVISION}.`);
  }
  const normalizedDataset = {
    repositoryId: DATASET_REPOSITORY_ID,
    revision: DATASET_REVISION,
  };
  const observation = record(manifest.observation, "manifest.observation");
  const policies = exactArray(manifest.policies, 2, "manifest.policies");
  policies.forEach((candidate, index) => {
    const policy = record(candidate, `manifest.policies[${index}]`);
    if (policy.policyId !== POLICY_IDS[index]) throw new CheckpointComparisonDataError(`manifest.policies[${index}].policyId has invalid order.`);
    if (policy.label !== POLICY_LABELS[index]) throw new CheckpointComparisonDataError(`manifest.policies[${index}].label is invalid.`);
  });
  const payloads = exactArray(manifest.payloads, 1, "manifest.payloads");
  const payload = record(payloads[0], "manifest.payloads[0]");
  if (payload.kind !== "plans" || payload.encoding !== "json" || typeof payload.filename !== "string") throw new CheckpointComparisonDataError("manifest plans payload is invalid.");
  if (!Number.isInteger(payload.byteSize) || (payload.byteSize as number) <= 0 || typeof payload.sha256 !== "string") throw new CheckpointComparisonDataError("manifest plans payload integrity metadata is invalid.");
  if (typeof observation.observationId !== "string" || !observation.observationId) throw new CheckpointComparisonDataError("manifest.observation.observationId must be non-empty.");
  return { ...manifest, dataset: normalizedDataset } as unknown as ComparisonManifest;
}

type ProjectedViolation = AvailableProjection["plans"][number]["jointLimitViolations"][number];

function violations(
  value: unknown,
  policyId: (typeof POLICY_IDS)[number],
  path: string,
): ProjectedViolation[] {
  if (!Array.isArray(value)) {
    throw new CheckpointComparisonDataError(`${path} must be an array.`);
  }
  const normalized = value.map((candidate, index): ProjectedViolation => {
    const itemPath = `${path}[${index}]`;
    const item = record(candidate, itemPath);
    exactFields(
      item,
      ["policyId", "stepIndex", "componentName", "urdfJointName", "value", "bound", "violationKind"],
      itemPath,
    );
    if (item.policyId !== policyId) {
      throw new CheckpointComparisonDataError(`${itemPath}.policyId must equal ${policyId}.`);
    }
    const stepIndex = integer(item.stepIndex, `${itemPath}.stepIndex`, 0, 49);
    if (typeof item.componentName !== "string" || !ARM_COMPONENTS.includes(item.componentName as (typeof ARM_COMPONENTS)[number])) {
      throw new CheckpointComparisonDataError(`${itemPath}.componentName must be a canonical arm-joint component.`);
    }
    const componentName = item.componentName as (typeof ARM_COMPONENTS)[number];
    const jointNumber = componentName.match(/_joint_([1-6])\.pos$/)?.[1];
    const expectedJoint = `joint${jointNumber}`;
    if (item.urdfJointName !== expectedJoint) {
      throw new CheckpointComparisonDataError(`${itemPath}.urdfJointName must equal ${expectedJoint} for ${componentName}.`);
    }
    const itemValue = finite(item.value, `${itemPath}.value`);
    const bound = finite(item.bound, `${itemPath}.bound`);
    if (item.violationKind !== "lower" && item.violationKind !== "upper") {
      throw new CheckpointComparisonDataError(`${itemPath}.violationKind must be lower or upper.`);
    }
    const violationKind = item.violationKind;
    return {
      policyId,
      stepIndex,
      componentName,
      urdfJointName: expectedJoint,
      value: itemValue,
      bound,
      violationKind,
    };
  });
  const key = (item: ProjectedViolation) => [
    item.stepIndex,
    ARM_COMPONENTS.indexOf(item.componentName as (typeof ARM_COMPONENTS)[number]),
    item.violationKind === "lower" ? 0 : 1,
  ] as const;
  for (let index = 1; index < normalized.length; index += 1) {
    const previous = key(normalized[index - 1]);
    const current = key(normalized[index]);
    const comparison = current[0] - previous[0] || current[1] - previous[1] || current[2] - previous[2];
    if (comparison === 0) {
      throw new CheckpointComparisonDataError(`${path}[${index}] duplicates the previous violation.`);
    }
    if (comparison < 0) {
      throw new CheckpointComparisonDataError(`${path}[${index}] is not in deterministic order.`);
    }
  }
  return normalized;
}

function decodeAvailableProjection(value: Record<string, unknown>, plans: ComparisonPlans["plans"]): AvailableProjection {
  if (value.sharedConfiguration !== true) throw new CheckpointComparisonDataError("trajectoryProjection.sharedConfiguration must be true.");
  const interpretation = record(value.actionInterpretation, "trajectoryProjection.actionInterpretation");
  if (
    interpretation.interpretationId !== "pi05-postprocessed-absolute-position-targets" ||
    interpretation.useRelativeActions !== false ||
    interpretation.deltaActionsPreprocessorEnabled !== false ||
    interpretation.absoluteActionsPostprocessorEnabled !== false ||
    interpretation.initialStateParticipatesInTransformation !== false
  ) throw new CheckpointComparisonDataError("trajectoryProjection action interpretation is unsupported.");
  const robot = record(value.robot, "trajectoryProjection.robot");
  if (robot.rotationRepresentation !== "quaternion" || JSON.stringify(robot.rotationComponentOrder) !== JSON.stringify(["X", "Y", "Z", "W"])) throw new CheckpointComparisonDataError("trajectoryProjection.robot quaternion order must be XYZW.");
  if (robot.calibratedGripperGeometry !== false) throw new CheckpointComparisonDataError("trajectoryProjection.robot.calibratedGripperGeometry must be false.");
  if (typeof robot.gripperSemanticDisclaimer !== "string" || !robot.gripperSemanticDisclaimer.includes("not physical jaw widths")) throw new CheckpointComparisonDataError("trajectoryProjection.robot must retain the raw-gripper disclaimer.");
  const targetLink = String(robot.targetLink ?? "");
  if (value.jointLimitPolicy !== "reject" && value.jointLimitPolicy !== "allow-with-recorded-violations") {
    throw new CheckpointComparisonDataError("trajectoryProjection.jointLimitPolicy is unsupported.");
  }
  const jointLimitPolicy = value.jointLimitPolicy;
  const projected = exactArray(value.plans, 2, "trajectoryProjection.plans");
  const normalizedProjected = projected.map((candidate, index) => {
    const projectedPlan = record(candidate, `trajectoryProjection.plans[${index}]`);
    if (projectedPlan.policyId !== POLICY_IDS[index]) throw new CheckpointComparisonDataError(`trajectoryProjection.plans[${index}].policyId has invalid order.`);
    const times = exactArray(projectedPlan.relativeTimesSeconds, 50, `trajectoryProjection.plans[${index}].relativeTimesSeconds`);
    times.forEach((time, step) => {
      if (finite(time, `trajectoryProjection.plans[${index}].relativeTimesSeconds[${step}]`) !== plans[index].relativeTimesSeconds[step]) throw new CheckpointComparisonDataError(`trajectoryProjection.plans[${index}].relativeTimesSeconds[${step}] must match the authoritative plan.`);
    });
    const left = arm(projectedPlan.left, "left", targetLink, `trajectoryProjection.plans[${index}].left`);
    const right = arm(projectedPlan.right, "right", targetLink, `trajectoryProjection.plans[${index}].right`);
    const normalizedViolations = violations(
      projectedPlan.jointLimitViolations,
      POLICY_IDS[index],
      `trajectoryProjection.plans[${index}].jointLimitViolations`,
    );
    if (jointLimitPolicy === "reject" && normalizedViolations.length > 0) {
      throw new CheckpointComparisonDataError(`trajectoryProjection.plans[${index}].jointLimitViolations must be empty when jointLimitPolicy is reject.`);
    }
    return {
      ...projectedPlan,
      policyId: POLICY_IDS[index],
      relativeTimesSeconds: [...plans[index].relativeTimesSeconds],
      left,
      right,
      jointLimitViolations: normalizedViolations,
    };
  });
  return { ...value, jointLimitPolicy, plans: normalizedProjected } as unknown as AvailableProjection;
}

export function decodeComparisonPlans(value: unknown): ComparisonPlans {
  const payload = record(value, "plans");
  schema(payload.schema, "plans.schema");
  if (payload.actionDimension !== 14 || payload.chunkLength !== 50) throw new CheckpointComparisonDataError("plans must use 50 × 14 actions.");
  const plans = exactArray(payload.plans, 2, "plans.plans").map((candidate, index) => {
    const plan = record(candidate, `plans.plans[${index}]`);
    if (plan.policyId !== POLICY_IDS[index]) throw new CheckpointComparisonDataError(`plans.plans[${index}].policyId has invalid order.`);
    const times = relativeTimes(
      plan.relativeTimesSeconds,
      `plans.plans[${index}].relativeTimesSeconds`,
    );
    return { policyId: POLICY_IDS[index], relativeTimesSeconds: times, actions: samples(plan.actions, 14, `plans.plans[${index}].actions`) };
  }) as ComparisonPlans["plans"];
  const projection = record(payload.trajectoryProjection, "trajectoryProjection");
  const normalizedProjection = projection.available === true
    ? decodeAvailableProjection(projection, plans)
    : projection;
  if (projection.available === false) {
    if (typeof projection.reason !== "string" || !projection.reason.trim()) throw new CheckpointComparisonDataError("trajectoryProjection.reason must be non-empty.");
    if (Object.keys(projection).some((key) => !["available", "reason"].includes(key))) throw new CheckpointComparisonDataError("unavailable trajectoryProjection must not contain fabricated trajectory data.");
  } else if (projection.available !== true) throw new CheckpointComparisonDataError("trajectoryProjection.available must be boolean.");
  return { ...payload, plans, trajectoryProjection: normalizedProjection } as unknown as ComparisonPlans;
}

export function decodeCheckpointComparison(manifestValue: unknown, plansValue: unknown): CheckpointComparisonData {
  const manifest = decodeComparisonManifest(manifestValue);
  const plans = decodeComparisonPlans(plansValue);
  if (manifest.observation.observationId !== plans.observationId) throw new CheckpointComparisonDataError("manifest and plans observation IDs must match.");
  return { manifest, plans };
}
