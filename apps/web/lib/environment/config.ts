import { validateLocalManifestPath } from "./path-safety";

export type LocalEnvironmentConfiguration =
  | { status: "unavailable" }
  | { status: "available"; manifestPath: string };

export function resolveLocalEnvironmentConfiguration(
  nodeEnv: string | undefined,
  configuredPath: string | undefined,
): LocalEnvironmentConfiguration {
  if (nodeEnv === "production" || !configuredPath) return { status: "unavailable" };
  try {
    return { status: "available", manifestPath: validateLocalManifestPath(configuredPath) };
  } catch {
    return { status: "unavailable" };
  }
}

export function isConservativeSpikeMobile(navigatorValue: Navigator): boolean {
  const nav = navigatorValue as Navigator & { userAgentData?: { mobile?: boolean } };
  if (nav.userAgentData?.mobile === true) return true;
  if (nav.platform === "MacIntel" && nav.maxTouchPoints > 1) return true;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent);
}

export function configuredLocalEnvironment(): LocalEnvironmentConfiguration {
  return resolveLocalEnvironmentConfiguration(
    process.env.NODE_ENV,
    process.env.NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST,
  );
}
