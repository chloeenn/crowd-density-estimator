import { useState, type ReactNode } from "react";

interface Props {
  stale?: boolean;
  originalUrl: string;
  heatmap: string;
  /** Photo with labelled object boxes, when the server detected objects. */
  objectsImage?: string;
  name: string;
  /** Drawn over the photo (the marked area). */
  overlay?: ReactNode;
  /** The colour scale; the opacity slider sits right under it. */
  legend?: ReactNode;
}

/** The returned heatmap is already blended with the photo. Stacking it over the original and
 * lowering its opacity fades back to the plain photo; the toggle flips between the two. */
type View = "heatmap" | "original" | "objects";

export function HeatmapViewer({ originalUrl, heatmap, objectsImage, name, stale, overlay, legend }: Props) {
  const views: View[] = objectsImage ? ["objects", "heatmap", "original"] : ["heatmap", "original"];
  // Objects is the default view whenever labelled boxes exist.
  const [picked, setView] = useState<View | null>(null);
  const view: View = picked && views.includes(picked) ? picked : views[0];
  const [opacity, setOpacity] = useState(100);
  const showHeat = view === "heatmap";

  return (
    <figure className="space-y-3">
      <div className={`relative overflow-hidden rounded-2xl transition-opacity duration-300 ease-out ${stale ? "opacity-50" : ""}`}>
        <img src={originalUrl} alt={`Uploaded photo: ${name}`} className="block w-full" />
        <img
          src={heatmap}
          alt={showHeat ? "Density heatmap: brighter colours mark areas with more people" : ""}
          aria-hidden={!showHeat}
          className="absolute inset-0 h-full w-full"
          style={{ opacity: showHeat ? opacity / 100 : 0 }}
        />
        {objectsImage && (
          <img
            src={objectsImage}
            alt={view === "objects" ? "Photo with labelled objects: each box is named" : ""}
            aria-hidden={view !== "objects"}
            className="absolute inset-0 h-full w-full"
            style={{ opacity: view === "objects" ? 1 : 0 }}
          />
        )}
        {overlay}
      </div>
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
        <div role="group" aria-label="Image view" className="flex gap-2">
          {views.map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className="pill pill-ghost"
            >
              {v === "heatmap" ? "Heatmap" : v === "original" ? "Original" : "Objects"}
            </button>
          ))}
        </div>
      </figcaption>
      {showHeat && legend}
      {showHeat && (
      <label className="flex items-center gap-3 text-sm text-muted">
        Heatmap opacity
        <input
          type="range"
          min={0}
          max={100}
          value={opacity}
          disabled={!showHeat}
          onChange={(e) => setOpacity(Number(e.target.value))}
          aria-valuetext={`${opacity}%`}
        />
        <span className="w-10 tabular-nums text-primary">{opacity}%</span>
      </label>
      )}
    </figure>
  );
}
