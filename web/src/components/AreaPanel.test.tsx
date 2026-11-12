import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AreaPanel } from "./AreaPanel";

const noop = () => {};
const base = { applied: false, drawing: false, onDrawing: noop, onRegion: noop, onApply: noop, onRemove: noop };
const square: [number, number][] = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];

describe("AreaPanel", () => {
  it("says what the number is for", () => {
    render(<AreaPanel {...base} value={{ region: [] }} />);
    expect(screen.getByText("people per m²", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText(/Low to Critical/)).toBeInTheDocument();
  });

  it("applies the area for the whole photo", () => {
    const onApply = vi.fn();
    render(<AreaPanel {...base} value={{ region: [] }} onApply={onApply} />);
    const button = screen.getByRole("button", { name: "Get people per m²" });
    expect(button).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("e.g. 200"), { target: { value: "200" } });
    fireEvent.click(button);
    expect(onApply).toHaveBeenCalledWith({ areaM2: 200, region: [] });
  });

  it("asks for an outline before the size when only part of the photo is meant", () => {
    const onApply = vi.fn();
    render(<AreaPanel {...base} value={{ region: [] }} onApply={onApply} />);
    fireEvent.click(screen.getByLabelText(/Part of the photo/));
    expect(screen.getByRole("button", { name: "Outline the area on the photo" })).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("e.g. 200"), { target: { value: "60" } });
    expect(screen.getByRole("button", { name: "Get people per m²" })).toBeDisabled();
    expect(screen.getByText("Outline the area first, then enter its size.")).toBeInTheDocument();
  });

  it("sends the outline with the size once it has three corners", () => {
    const onApply = vi.fn();
    render(<AreaPanel {...base} value={{ region: square, areaM2: 60 }} onApply={onApply} />);
    expect(screen.getByText("Size of the outlined area (m²)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Get people per m²" }));
    expect(onApply).toHaveBeenCalledWith({ areaM2: 60, region: square });
  });

  it("counts corners while outlining and needs three to finish", () => {
    render(<AreaPanel {...base} drawing value={{ region: [[0.1, 0.1], [0.5, 0.5]] }} />);
    expect(screen.getByRole("status")).toHaveTextContent("2 placed, at least 3 needed");
    expect(screen.getByRole("button", { name: "Done outlining" })).toBeDisabled();
  });
});
