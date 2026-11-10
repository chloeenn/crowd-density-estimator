import { useState } from "react";

interface Props {
  originalUrl: string;
  heatmap: string;
  name: string;
}

/** The returned heatmap is already blended with the photo. Stacking it over the original and
 * lowering its opacity fades back to the plain photo; the toggle flips between the two. */
export function HeatmapViewer({ originalUrl, heatmap, name }: Props) {
  const [view, setView] = useState<"heatmap" | "original">("heatmap");
  const [opacity, setOpacity] = useState(100);
  const showHeat = view === "heatmap";

  return (
    <figure className="space-y-3">
      <div className="relative overflow-hidden rounded-xl bg-slate-900">
        <img src={originalUrl} alt={`Uploaded photo: ${name}`} className="block w-full" />
        <img
          src={heatmap}
          alt="Density heatmap: warmer colours mark areas with more people"
          className="absolute inset-0 h-full w-full"
          style={{ opacity: showHeat ? opacity / 100 : 0 }}
        />
      </div>
      <figcaption className="flex flex-wrap items-center gap-4">
        <div role="group" aria-label="Image view" className="inline-flex overflow-hidden rounded-lg border border-slate-400">
          {(["heatmap", "original"] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 text-sm font-medium focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-400 ${
                view === v ? "bg-indigo-700 text-white" : "bg-white text-slate-800 hover:bg-slate-100"
              }`}
            >
              {v === "heatmap" ? "Heatmap" : "Original"}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-800">
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
          <span className="w-10 tabular-nums">{opacity}%</span>
        </label>
      </figcaption>
    </figure>
  );
}
