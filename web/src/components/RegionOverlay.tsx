import type { Point } from "../api/types";

/** The marked area drawn over the photo. While `drawing`, a click adds a corner (x, y as 0-1 fractions). */
export function RegionOverlay({ points, drawing, onAdd }: { points: Point[]; drawing: boolean; onAdd: (p: Point) => void }) {
  if (!drawing && points.length === 0) return null;
  const path = points.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      role={drawing ? "application" : "img"}
      aria-label={drawing ? "Click the photo to mark the corners of the area" : "Marked area"}
      className={`absolute inset-0 h-full w-full ${drawing ? "cursor-crosshair" : "pointer-events-none"}`}
      onClick={(e) => {
        if (!drawing) return;
        const r = e.currentTarget.getBoundingClientRect();
        const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
        const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
        onAdd([Number(x.toFixed(4)), Number(y.toFixed(4))]);
      }}
    >
      {points.length >= 3 ? (
        <polygon points={path} fill="rgba(255,255,255,0.18)" stroke="#fff" strokeWidth="0.6" vectorEffect="non-scaling-stroke" />
      ) : (
        <polyline points={path} fill="none" stroke="#fff" strokeWidth="0.6" vectorEffect="non-scaling-stroke" />
      )}
      {points.map(([x, y], i) => (
        <circle key={i} cx={x * 100} cy={y * 100} r="0.9" fill="#fff" stroke="#32302f" strokeWidth="0.3" vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}
