"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { AtlasDataProvider } from "./AtlasDataProvider";
import { ViewerStore } from "./ViewerStore";

function ViewerLoading() {
  return (
    <div className="viewer-status" role="status" aria-live="polite">
      <span className="status-pulse" aria-hidden="true" />
      Loading 3D viewer…
    </div>
  );
}

const AtlasViewer = dynamic(
  () => import("./AtlasViewer").then((module) => module.AtlasViewer),
  {
    ssr: false,
    loading: ViewerLoading,
  },
);

function hasWebGL() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGL2RenderingContext && canvas.getContext("webgl2"),
    );
  } catch {
    return false;
  }
}

export function ViewerEntry() {
  const [support, setSupport] = useState<"checking" | "ready" | "missing">(
    "checking",
  );

  useEffect(() => {
    const detectionFrame = window.requestAnimationFrame(() => {
      setSupport(hasWebGL() ? "ready" : "missing");
    });

    return () => window.cancelAnimationFrame(detectionFrame);
  }, []);

  if (support === "checking") {
    return <ViewerLoading />;
  }

  if (support === "missing") {
    return (
      <section className="viewer-fallback" role="alert">
        <p className="eyebrow">WebGL unavailable</p>
        <h1>The interactive scene needs WebGL 2.</h1>
        <p>
          This browser or graphics configuration cannot create the viewer. The
          methodology and source data remain available without WebGL.
        </p>
        <a className="button button-secondary" href="/methodology">
          Read the methodology
        </a>
      </section>
    );
  }
  return (
    <AtlasDataProvider>
      <ViewerStore>
        <AtlasViewer />
      </ViewerStore>
    </AtlasDataProvider>
  );
}
