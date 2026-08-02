"use client";

import { useEffect, useState } from "react";

import { loadCheckpointComparison } from "@/lib/checkpoint-comparison/load";
import type { CheckpointComparisonData } from "@/lib/checkpoint-comparison/types";

import { ComparisonViewer } from "./ComparisonViewer";

export function ComparisonEntry() {
  const [state, setState] = useState<{ data?: CheckpointComparisonData; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    const bundle = new URLSearchParams(window.location.search).get("bundle");
    if (!bundle) {
      const frame = window.requestAnimationFrame(() => setState({ loading: false }));
      return () => window.cancelAnimationFrame(frame);
    }
    const controller = new AbortController();
    loadCheckpointComparison(bundle, controller.signal).then((data) => setState({ data, loading: false })).catch((error) => {
      if (!controller.signal.aborted) setState({ error: String(error instanceof Error ? error.message : error), loading: false });
    });
    return () => controller.abort();
  }, []);
  if (state.loading) return <div className="viewer-status" role="status">Loading checkpoint comparison…</div>;
  if (state.error) return <section className="viewer-fallback" role="alert"><p className="eyebrow">Comparison data error</p><h1>The generated comparison bundle could not be loaded.</h1><p>{state.error}</p></section>;
  if (!state.data) return <section className="viewer-fallback"><p className="eyebrow">No comparison artifact configured</p><h1>Base π0.5 vs Fine-tuned π0.5</h1><p>This viewer requires a validated checkpoint-comparison v1.1 bundle. No synthetic result is presented as real policy inference.</p><p>Provide a safe same-origin <code>?bundle=/path/to/comparison</code> URL after generating a comparison artifact.</p></section>;
  return <ComparisonViewer data={state.data} />;
}
