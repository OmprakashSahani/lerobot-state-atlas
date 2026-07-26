import type { Metadata } from "next";

import { ViewerEntry } from "@/components/viewer/ViewerEntry";

export const metadata: Metadata = { title: "Demo viewer" };

export default function DemoViewerPage() {
  return <ViewerEntry />;
}
