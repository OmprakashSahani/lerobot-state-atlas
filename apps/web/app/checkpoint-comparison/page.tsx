import type { Metadata } from "next";

import { ComparisonEntry } from "@/components/checkpoint-comparison/ComparisonEntry";

export const metadata: Metadata = {
  title: "Checkpoint comparison",
  description: "Inspect generated Base π0.5 and Fine-tuned π0.5 policy plans.",
};

export default function CheckpointComparisonPage() {
  return <ComparisonEntry />;
}
