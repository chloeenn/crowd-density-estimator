import type { Level } from "../api/types";

// Colour is never the only signal: every level has its own icon and a text label.
const STYLE: Record<Level, { icon: string; cls: string }> = {
  Low: { icon: "✓", cls: "bg-emerald-100 text-emerald-900 border-emerald-700" },
  Medium: { icon: "●", cls: "bg-yellow-100 text-yellow-900 border-yellow-700" },
  High: { icon: "▲", cls: "bg-orange-100 text-orange-900 border-orange-700" },
  Critical: { icon: "⚠", cls: "bg-red-100 text-red-900 border-red-700" },
};

export function LevelBadge({ level }: { level: Level }) {
  const { icon, cls } = STYLE[level];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-semibold ${cls}`}>
      <span aria-hidden="true">{icon}</span>
      {level} density
    </span>
  );
}
