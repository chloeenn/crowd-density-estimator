import type { DensityBands, Level, Thresholds } from "../api/types";
import { LEVELS, SIZES, densityRange, rangeFor } from "../status";

/** The four levels with the range each covers: people per m² when an area is known, otherwise the head count.
 * The band that matches the current image is marked in words ("This image"), not just by tint. */
export function ReferenceStrip({ thresholds, densityBands, level, byArea }: { thresholds?: Thresholds; densityBands?: DensityBands; level?: Level; byArea: boolean }) {
  return (
    <section aria-label={byArea ? "Density levels" : "Crowd sizes"}>
      <dl className="grid grid-cols-2 gap-x-8 gap-y-10 lg:grid-cols-4">
        {LEVELS.map((l) => {
          const current = level === l.id;
          return (
            <div key={l.id} aria-current={current || undefined} className={`space-y-2 ${level && !current ? "opacity-60" : ""}`}>
              <dt className="flex items-baseline">
                <span className="font-display text-xl font-semibold text-primary">{byArea ? l.label : SIZES[l.id].label}</span>
              </dt>
              <dd className="text-sm tabular-nums text-primary">
                {byArea ? densityRange(l.id, densityBands) : rangeFor(l.id, thresholds)}
                <span className="ml-1 text-muted">{byArea ? "people per m²" : "people"}</span>
              </dd>
              <dd className="text-sm leading-relaxed text-muted">{byArea ? l.description : SIZES[l.id].description}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
