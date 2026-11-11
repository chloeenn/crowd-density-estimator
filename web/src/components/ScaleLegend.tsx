/** The stops of ml/crowd/viz.py's LEGEND_STOPS (viridis), sparse to dense. ml/tests/test_viz.py fails if the API's
 * colormap changes without them, so the legend cannot drift from the heatmap it describes. */
const GRADIENT = "linear-gradient(to right, #440154, #414487, #2a788e, #22a884, #7ad151, #fde725)";

export function ScaleLegend() {
  return (
    <div className="space-y-2" role="group" aria-label="Heatmap colour scale">
      <div
        role="img"
        aria-label="Colour scale from dark purple for the fewest people, through blue and green, to yellow for the most people in this image"
        className="h-2.5 w-full rounded-full"
        style={{ background: GRADIENT }}
      />
      <div className="flex justify-between text-xs text-muted">
        <span>Fewer</span>
        <span>More</span>
      </div>
    </div>
  );
}
