export type RuntimeEnvironment = "development" | "production" | "test";

export function atlasCacheControl(environment: RuntimeEnvironment): string {
  return environment === "development"
    ? "no-store, max-age=0"
    : "public, max-age=31536000, immutable";
}

export function atlasFetchOptions(
  environment: RuntimeEnvironment,
): RequestInit | undefined {
  return environment === "development" ? { cache: "no-store" } : undefined;
}
