import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";
import { LevelBadge } from "./components/LevelBadge";

vi.mock("./api/client", async (orig) => ({
  ...(await orig<object>()),
  getMetadata: vi.fn().mockResolvedValue({
    default_part: "A",
    models: [
      { part: "A", version: "a", git_sha: "x", run_id: "r", epoch: 1, val_mae: null, val_rmse: null },
      { part: "B", version: "b", git_sha: "x", run_id: "r", epoch: 1, val_mae: null, val_rmse: null },
    ],
    thresholds: { low: 50, medium: 150, high: 400, critical: 800 },
    density_bands: { medium: 1, high: 2, critical: 4 },
    routing_threshold: 150,
    limits: { max_upload_mb: 10, allowed_content_types: ["image/jpeg"] },
  }),
}));

describe("App", () => {
  it("renders an accessible dropzone, captioned samples and the model switch", async () => {
    render(<App />);
    expect(screen.getByRole("button", { name: /upload a crowd photo/i })).toHaveAttribute("tabindex", "0");
    // Samples carry a visible caption as their accessible name; the thumbnails themselves are decorative.
    expect(screen.getByRole("button", { name: "Street scene 1" })).toBeInTheDocument();
    expect(await screen.findByLabelText(/Sparse/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Automatic/)).toBeChecked();
  });
});

describe("LevelBadge", () => {
  it.each([["Low", "Small crowd"], ["Critical", "Very large crowd"]] as const)("names the %s count band %s", (level, label) => {
    render(<LevelBadge level={level} byArea={false} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it.each(["Low", "Medium", "High", "Critical"] as const)("shows %s as a text label", (level) => {
    render(<LevelBadge level={level} />);
    expect(screen.getByText(new RegExp(`${level} density`))).toBeInTheDocument();
  });
});
