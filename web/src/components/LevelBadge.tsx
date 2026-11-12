import type { Level } from "../api/types";
import { SIZES, levelStyle } from "../status";
import { Pips } from "./Pips";

// Severity reads from fill weight and a four-pip meter, plus the text label, so colour is never the only signal.
export function LevelBadge({ level, byArea = true }: { level: Level; byArea?: boolean }) {
  const { cls, label: densityLabel } = levelStyle(level);
  const label = byArea ? densityLabel : SIZES[level].label;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium ${cls}`}>
      <Pips level={level} />
      {label}
    </span>
  );
}
