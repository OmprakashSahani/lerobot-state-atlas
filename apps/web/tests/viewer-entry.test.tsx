import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ViewerEntry } from "@/components/viewer/ViewerEntry";

describe("viewer client boundary", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps stable loading markup until mounted WebGL detection completes", () => {
    let detect: FrameRequestCallback | undefined;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      detect = callback;
      return 1;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});

    render(<ViewerEntry />);

    expect(screen.getByRole("status")).toHaveTextContent("Loading 3D viewer");

    act(() => detect?.(0));

    expect(screen.getByRole("alert")).toHaveTextContent("WebGL unavailable");
    expect(
      screen.getByRole("link", { name: "Read the methodology" }),
    ).toHaveAttribute("href", "/methodology");
  });
});
