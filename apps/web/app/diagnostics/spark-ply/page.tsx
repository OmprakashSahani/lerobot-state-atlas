import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { IsolatedSparkPlyDiagnostic } from "@/components/viewer/IsolatedSparkPlyDiagnostic";

export const metadata: Metadata = { title: "Exact Spark source comparison" };

export default function IsolatedSparkPlyDiagnosticPage() {
  if (process.env.NODE_ENV !== "development") notFound();
  return <IsolatedSparkPlyDiagnostic />;
}
