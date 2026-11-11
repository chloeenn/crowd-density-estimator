import type { Level } from "../api/types";
import { LEVELS, levelRank, levelStyle } from "../status";

/** Severity as quantity: four pips, filled up to the level, in the level's heatmap colour. Text always names the level too. */
export function Pips({ level }: { level: Level }) {
  const rank = levelRank(level);
  const { ramp } = levelStyle(level);
  return (
    <span aria-hidden="true" className="inline-flex gap-0.5">
      {LEVELS.map((l, i) => (
        <span key={l.id} className={`h-2.5 w-2.5 rounded-full ${i < rank ? ramp : "bg-current opacity-25"}`} />
      ))}
    </span>
  );
}
