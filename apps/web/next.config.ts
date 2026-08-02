import type { NextConfig } from "next";

import {
  atlasCacheControl,
  type RuntimeEnvironment,
} from "./lib/data/cachePolicy";

const isDevelopment = process.env.NODE_ENV === "development";
const isVercelPreview = process.env.VERCEL_ENV === "preview";
const runtimeEnvironment = process.env.NODE_ENV as RuntimeEnvironment;

const scriptSources = ["'self'", "'unsafe-inline'"];
const styleSources = ["'self'", "'unsafe-inline'"];
const imageSources = ["'self'", "data:", "blob:"];
const fontSources = ["'self'"];
const connectSources = ["'self'"];
const frameSources = isVercelPreview
  ? ["https://vercel.live"]
  : ["'none'"];

if (isDevelopment) {
  scriptSources.push("'unsafe-eval'");
  // Spark 2.1.0 initializes embedded WASM through fetch(data:...) in the local-only spike.
  connectSources.push("data:");
}

if (isVercelPreview) {
  scriptSources.push("https://vercel.live");
  styleSources.push("https://vercel.live");
  imageSources.push("https://vercel.live", "https://vercel.com");
  fontSources.push("https://vercel.live", "https://assets.vercel.com");
  connectSources.push(
    "https://vercel.live",
    "wss://ws-us3.pusher.com",
  );
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
      `style-src ${styleSources.join(" ")}`,
      `img-src ${imageSources.join(" ")}`,
      `font-src ${fontSources.join(" ")}`,
      `connect-src ${connectSources.join(" ")}`,
      "worker-src 'self' blob:",
      `frame-src ${frameSources.join(" ")}`,
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
