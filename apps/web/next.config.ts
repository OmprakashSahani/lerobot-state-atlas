import type { NextConfig } from "next";

import {
  atlasCacheControl,
  type RuntimeEnvironment,
} from "./lib/data/cachePolicy";

const isDevelopment = process.env.NODE_ENV === "development";
const runtimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment;
const scriptSources = ["'self'", "'unsafe-inline'"];

if (isDevelopment) {
  scriptSources.push("'unsafe-eval'");
}

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=()",
  },
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      `script-src ${scriptSources.join(" ")}`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self'",
      "connect-src 'self'",
      "worker-src 'self' blob:",
      "object-src 'none'",
      "base-uri 'self'",
      "frame-ancestors 'none'",
    ].join("; "),
  },
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
      {
        source: "/atlas-data/:version/:asset*",
        headers: [
          {
            key: "Cache-Control",
            value: atlasCacheControl(runtimeEnvironment),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
