import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EnvironmentStatus } from "@/components/viewer/EnvironmentStatus";
import { demoEnvironmentCapability } from "@/lib/environment/types";

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
});
