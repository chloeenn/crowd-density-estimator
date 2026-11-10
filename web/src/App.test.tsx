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
    level_caveat: "caveat",
    limits: { max_upload_mb: 10, allowed_content_types: ["image/jpeg"] },
  }),
}));

describe("App", () => {
  it("renders an accessible dropzone, sample alt text and the model switch", async () => {
    render(<App />);
    expect(screen.getByRole("button", { name: /upload a crowd photo/i })).toHaveAttribute("tabindex", "0");
    for (const img of screen.getAllByRole("img")) expect(img).toHaveAttribute("alt");
    expect(await screen.findByLabelText(/Part B/)).toBeInTheDocument();
  });
});

describe("LevelBadge", () => {
  it.each(["Low", "Medium", "High", "Critical"] as const)("shows %s as text and an icon", (level) => {
    render(<LevelBadge level={level} />);
    expect(screen.getByText(new RegExp(`${level} density`))).toBeInTheDocument();
  });
});
