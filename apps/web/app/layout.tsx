import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "LeRobot State Atlas",
    template: "%s · LeRobot State Atlas",
  },
  description:
    "A shared-world technical viewer for LeRobot dual-arm workspace coverage.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <header className="site-header">
          <Link className="brand" href="/" aria-label="LeRobot State Atlas home">
            <span className="brand-mark" aria-hidden="true" />
            <span>LeRobot State Atlas</span>
          </Link>
          <nav aria-label="Primary navigation">
            <Link href="/checkpoint-comparison">Checkpoint comparison</Link>
            <Link href="/methodology">Methodology</Link>
            <Link href="/capture-guide">Capture guide</Link>
            <a
              href="https://github.com/OmprakashSahani/lerobot-state-atlas"
              rel="noreferrer"
              target="_blank"
            >
              GitHub
            </a>
          </nav>
        </header>
        <main id="main-content">{children}</main>
      </body>
    </html>
  );
}
