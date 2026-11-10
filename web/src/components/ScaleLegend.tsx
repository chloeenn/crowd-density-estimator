import type { PredictResponse } from "../api/types";

export function ScaleLegend({ result }: { result: PredictResponse }) {
  const max = result.max_concentration;
  return (
    <div className="space-y-1" role="group" aria-label="Heatmap colour scale">
      <div
        className="h-3 w-full rounded"
        style={{ background: "linear-gradient(to right, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000)" }}
        aria-hidden="true"
      />
      <div className="flex justify-between text-xs tabular-nums text-slate-700">
        <span>0 (fewer people)</span>
        <span>{max.toFixed(2)} (most, this image)</span>
      </div>
      <p className="text-xs text-slate-600">
        Scale is people per {result.concentration_basis.cell_input_px}×{result.concentration_basis.cell_input_px} pixel
        cell in the image, normalised per photo. It is not people per square metre.
      </p>
    </div>
  );
}
