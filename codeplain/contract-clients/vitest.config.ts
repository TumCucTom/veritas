// Runs only the canonical generated conformance suite (build_conformance_tests/)
// against the canonical generated package (build/src). The Codeplain build also
// leaves verbatim duplicate trees (plain_modules/, conformance_tests/) and a
// jest-globals unit suite under build/src; those are excluded here so a single
// `npx vitest run` is deterministic and green.
//
// A plain config object is exported (instead of importing `defineConfig` from
// `vitest/config`) so the file loads even when vitest is run via `npx` without a
// local node_modules.
export default {
  test: {
    environment: "node",
    include: ["build_conformance_tests/**/*.test.ts"],
    exclude: [
      "**/node_modules/**",
      "build/**",
      "plain_modules/**",
      "conformance_tests/**",
      "dist/**",
      "dist_conformance_tests/**",
    ],
  },
};
