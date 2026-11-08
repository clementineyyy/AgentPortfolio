import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack(config) {
    // CopilotKit treats MapLibre as an optional peer and falls back to a places list without it.
    config.resolve.alias = {
      ...config.resolve.alias,
      "maplibre-gl": false,
    };
    return config;
  },
};

export default nextConfig;
