import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { EnvironmentStatus } from "@/components/viewer/EnvironmentStatus";
import { demoEnvironmentCapability } from "@/lib/environment/types";
import type { LocalEnvironmentController } from "@/lib/environment/use-local-environment";

function local(phase: LocalEnvironmentController["phase"]): LocalEnvironmentController {
  return {
    phase,
    request: null,
    load: vi.fn(), hide: vi.fn(), show: vi.fn(), unload: vi.fn(), retry: vi.fn(),
    onRendererPhase: vi.fn(), onRendererError: vi.fn(), setWebGl2Supported: vi.fn(),
    ...(phase === "error" ? { error: "Synthetic parser rejected the asset." } : {}),
  };
}

describe("environment capability status", () => {
  it("presents intentional unavailability without a failure state", () => {
    render(<EnvironmentStatus capability={demoEnvironmentCapability} />);

    const section = screen.getByRole("region", { name: "Environment" });
    expect(within(section).getByText("Analytical grid active")).toBeVisible();
    expect(within(section).getByText("Unavailable")).toBeVisible();
    expect(within(section).getByRole("note")).toHaveTextContent(
      "No validated Gaussian Splat scan or environment-to-robot calibration is bundled with this demo.",
    );
    expect(within(section).getByRole("note")).toHaveTextContent(
      "No real reconstruction or calibrated environment alignment is claimed.",
    );
    expect(within(section).getByRole("note")).toHaveTextContent(
      "The robot workspace viewer remains fully available.",
    );
    expect(within(section).queryByRole("alert")).not.toBeInTheDocument();
    expect(within(section).queryByRole("button")).not.toBeInTheDocument();
  });

  it("exposes truthful local idle, error, and mobile refusal states", () => {
    const { rerender } = render(<EnvironmentStatus capability={demoEnvironmentCapability} local={local("idle")} />);
    expect(screen.getByText("Synthetic test environment — not a real reconstruction")).toBeVisible();
    expect(screen.getByRole("button", { name: "Load synthetic environment" })).toBeEnabled();
    rerender(<EnvironmentStatus capability={demoEnvironmentCapability} local={local("error")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("parser rejected");
    expect(screen.getByRole("button", { name: "Retry" })).toBeVisible();
    rerender(<EnvironmentStatus capability={demoEnvironmentCapability} local={local("mobile-refusal")} />);
    expect(screen.getByText(/intentionally disabled on mobile/i)).toBeVisible();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
