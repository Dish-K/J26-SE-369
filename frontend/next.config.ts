import type { NextConfig } from "next";

const backend = process.env.TELEMETRY_BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // The browser calls its own site, so the backend needs no CORS settings.
  async rewrites() {
    return [{ source: "/api/telemetry/:path*", destination: `${backend}/telemetry/:path*` }];
  },
};

export default nextConfig;
