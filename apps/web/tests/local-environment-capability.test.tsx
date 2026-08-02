import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useLocalEnvironmentSpike } from "@/lib/environment/use-local-environment";

const approved = "/environment-data/__local-synthetic__/manifest.json";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("local environment request gating", () => {
  it("does not request anything before explicit load", () => {
    vi.stubEnv("NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST", approved);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const { result } = renderHook(() => useLocalEnvironmentSpike());
    expect(result.current.phase).toBe("idle");
    expect(result.current.request).toBeNull();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("refuses mobile before manifest request or render request", () => {
    vi.stubEnv("NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST", approved);
    vi.stubGlobal("navigator", { userAgent: "Mozilla/5.0 Android Mobile", userAgentData: { mobile: true } });
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const { result } = renderHook(() => useLocalEnvironmentSpike());
    expect(result.current.phase).toBe("mobile-refusal");
    act(() => result.current.load());
    expect(fetcher).not.toHaveBeenCalled();
    expect(result.current.request).toBeNull();
  });

  it("refuses unsupported WebGL2 before a request", () => {
    vi.stubEnv("NEXT_PUBLIC_LOCAL_ENVIRONMENT_MANIFEST", approved);
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const { result } = renderHook(() => useLocalEnvironmentSpike());
    act(() => result.current.setWebGl2Supported(false));
    act(() => result.current.load());
    expect(result.current.phase).toBe("unsupported-webgl2");
    expect(fetcher).not.toHaveBeenCalled();
  });
});
