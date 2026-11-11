import type { Level, Thresholds } from "../api/types";
import { LEVELS, rangeFor } from "../status";

/** The same four levels as the badge, with the head-count range each one covers. The band that matches the
 * current image is marked in words ("This image"), not just by tint. */
export function ReferenceStrip({ thresholds, level }: { thresholds?: Thresholds; level?: Level }) {
  return (
    <section aria-label="Density levels">
      <dl className="grid grid-cols-2 gap-x-8 gap-y-10 lg:grid-cols-4">
        {LEVELS.map((l) => {
          const current = level === l.id;
          return (
            <div key={l.id} aria-current={current || undefined} className={`space-y-2 ${level && !current ? "opacity-60" : ""}`}>
              <dt className="flex items-baseline">
                <span className="font-display text-xl font-semibold text-primary">{l.label}</span>
              </dt>
              <dd className="text-sm tabular-nums text-primary">
                {rangeFor(l.id, thresholds)}
                <span className="ml-1 text-muted">people</span>
              </dd>
              <dd className="text-sm leading-relaxed text-muted">{l.description}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
