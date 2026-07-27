import { describe, expect, it } from "vitest";

import {
  atlasCacheControl,
  atlasFetchOptions,
} from "@/lib/data/cachePolicy";

describe("atlas cache policy", () => {
  it("disables storage for development responses and fetches", () => {
    expect(atlasCacheControl("development")).toBe("no-store, max-age=0");
    expect(atlasFetchOptions("development")).toEqual({ cache: "no-store" });
  });

  it("keeps production responses immutable without disabling fetch caching", () => {
    expect(atlasCacheControl("production")).toBe(
      "public, max-age=31536000, immutable",
    );
    expect(atlasFetchOptions("production")).toBeUndefined();
  });
});
