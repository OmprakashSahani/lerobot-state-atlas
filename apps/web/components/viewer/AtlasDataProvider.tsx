"use client";

import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";

import type { AtlasData } from "@/lib/atlas-schema/types";
import {
  benchmarksEnabled,
  publishBrowserBenchmarkIfEnabled,
  runAtlasBrowserBenchmark,
} from "@/lib/data/browserBenchmark";
import {
  loadDemoBundle,
  loadDemoBundleForBenchmark,
  resolveBundleBase,
} from "@/lib/data/loadBundle";

type AtlasDataState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: AtlasData };

const AtlasDataContext = createContext<AtlasDataState | null>(null);

export function AtlasDataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AtlasDataState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    const loadBenchmarkedBundle = () => {
      const bundleBase = resolveBundleBase();
      return publishBrowserBenchmarkIfEnabled(
        true,
        bundleBase,
        window,
        async () => {
          const startedAt = new Date();
          const result = await loadDemoBundleForBenchmark();
          return {
            data: result.data,
            report: runAtlasBrowserBenchmark(result.data, {
              bundleBase,
              loadDurations: result.durations,
              startedAtUtc: startedAt.toISOString(),
            }),
          };
        },
      )!;
    };
    const request = benchmarksEnabled()
      ? Promise.resolve().then(loadBenchmarkedBundle)
      : loadDemoBundle();
    request
      .then((data) => {
        if (active) setState({ status: "ready", data });
      })
      .catch((error: unknown) => {
        if (active) {
          setState({
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "The atlas data could not be loaded.",
          });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <AtlasDataContext.Provider value={state}>
      {children}
    </AtlasDataContext.Provider>
  );
}

export function useAtlasData() {
  const value = useContext(AtlasDataContext);
  if (!value) {
    throw new Error("useAtlasData must be used inside AtlasDataProvider.");
  }
  return value;
}
