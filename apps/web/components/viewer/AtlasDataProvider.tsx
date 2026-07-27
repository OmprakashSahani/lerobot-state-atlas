"use client";

import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";

import type { AtlasData } from "@/lib/atlas-schema/types";
import { loadDemoBundle } from "@/lib/data/loadBundle";

type AtlasDataState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; data: AtlasData };

const AtlasDataContext = createContext<AtlasDataState | null>(null);

export function AtlasDataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AtlasDataState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    loadDemoBundle()
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
