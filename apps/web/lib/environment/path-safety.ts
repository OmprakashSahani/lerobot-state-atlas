import {
  LOCAL_ENVIRONMENT_ROOT,
  LOCAL_REAL_ENVIRONMENT_ROOT,
} from "./limits";

export class EnvironmentPathError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EnvironmentPathError";
  }
}

const APPROVED_LOCAL_ENVIRONMENT_ROOTS = [
  LOCAL_ENVIRONMENT_ROOT,
  LOCAL_REAL_ENVIRONMENT_ROOT,
] as const;

function approvedRootForPath(path: string): string | undefined {
  return APPROVED_LOCAL_ENVIRONMENT_ROOTS.find(
    (root) => path.startsWith(root) && path !== root,
  );
}

function rejectEncodedHazards(path: string): void {
  let decoded: string;
  try {
    decoded = decodeURIComponent(path);
  } catch {
    throw new EnvironmentPathError(
      "Environment path has invalid percent encoding.",
    );
  }
  if (decoded !== path || /[\\?#\u0000-\u001f\u007f]/.test(decoded)) {
    throw new EnvironmentPathError(
      "Environment path contains encoded or unsafe characters.",
    );
  }
}

export function validateLocalManifestPath(path: string): string {
  if (
    typeof path !== "string" ||
    path.trim() !== path ||
    !approvedRootForPath(path) ||
    path.includes("//") ||
    /[\\?#\u0000-\u001f\u007f]/.test(path)
  ) {
    throw new EnvironmentPathError(
      "Local environment manifest path is outside the approved roots.",
    );
  }

  rejectEncodedHazards(path);

  const segments = path.slice(1).split("/");
  if (
    segments.some(
      (segment) => !segment || segment === "." || segment === "..",
    )
  ) {
    throw new EnvironmentPathError(
      "Local environment manifest path is not canonical.",
    );
  }

  if (segments.at(-1) !== "manifest.json") {
    throw new EnvironmentPathError(
      "Local environment path must name manifest.json.",
    );
  }

  return path;
}

export function resolveLocalAssetPath(
  manifestPath: string,
  filename: string,
): string {
  const canonicalManifest = validateLocalManifestPath(manifestPath);

  if (
    !/^[A-Za-z0-9][A-Za-z0-9._-]*\.spz$/.test(filename) ||
    filename.includes("..")
  ) {
    throw new EnvironmentPathError(
      "Environment asset filename is unsafe.",
    );
  }

  const slash = canonicalManifest.lastIndexOf("/");
  return `${canonicalManifest.slice(0, slash + 1)}${filename}`;
}

export function assertSafeFinalResponseUrl(
  response: Pick<Response, "redirected" | "url">,
  requestedPath: string,
  origin: string,
): void {
  if (response.redirected) {
    throw new EnvironmentPathError(
      "Environment redirects are not permitted.",
    );
  }

  const requestedRoot = approvedRootForPath(requestedPath);
  if (!requestedRoot) {
    throw new EnvironmentPathError(
      "Environment request path is outside the approved roots.",
    );
  }

  const expected = new URL(requestedPath, origin);
  const final = new URL(response.url || expected.href, origin);

  if (
    final.origin !== origin ||
    final.pathname !== expected.pathname ||
    final.search ||
    final.hash ||
    !final.pathname.startsWith(requestedRoot)
  ) {
    throw new EnvironmentPathError(
      "Environment response URL is outside the approved root.",
    );
  }
}
