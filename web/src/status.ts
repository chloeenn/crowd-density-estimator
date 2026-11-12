import type { DensityBands, Level, Part, Thresholds } from "./api/types";

/** One scale for the whole page: the four levels the API reports. Descriptions say what the band looks like,
 * never what to do about it: the model has no measured accuracy, so it must not give operational advice. */
export const LEVELS: {
  id: Level;
  label: string;
  description: string;
  cls: string; // surface weight: severity gets heavier, hue stays with the pips and the heatmap
  ramp: string; // solid swatch from the heatmap's colour ramp
}[] = [
  { id: "Low", label: "Low density", description: "Sparse. People are well spread out.", cls: "bg-surface text-primary border-[0.5px] border-edge", ramp: "bg-ramp-0" },
  { id: "Medium", label: "Medium density", description: "Moderate. Groups are starting to form.", cls: "bg-surface text-primary border border-edge", ramp: "bg-ramp-1" },
  { id: "High", label: "High density", description: "Dense. Little space between people.", cls: "bg-weight text-primary border border-primary", ramp: "bg-ramp-2" },
  { id: "Critical", label: "Critical density", description: "Very dense. People are packed closely together.", cls: "bg-primary text-page border border-primary", ramp: "bg-ramp-3" },
];

/** Without a ground area the level only compares the head count with the training scenes. It says how big the
 * crowd is, not how tightly it is packed, so it is named that way. */
export const SIZES: Record<Level, { label: string; description: string }> = {
  Low: { label: "Small crowd", description: "A few dozen to about 150 people." },
  Medium: { label: "Medium crowd", description: "Roughly 150 to 350 people." },
  High: { label: "Large crowd", description: "Roughly 350 to 700 people." },
  Critical: { label: "Very large crowd", description: "About 700 people or more." },
};

/** 1-4: how many of the four severity pips a level fills. */
export function levelRank(level: Level): number {
  return Math.max(0, LEVELS.findIndex((l) => l.id === level)) + 1;
}

export function levelStyle(level: Level) {
  return LEVELS.find((l) => l.id === level) ?? LEVELS[0];
}

/** Head-count range for each level, derived from the API's band edges (each edge is where a band starts). */
export function rangeFor(level: Level, t?: Thresholds): string {
  if (!t) return "—";
  const n = (v: number) => Math.round(v).toLocaleString();
  if (level === "Low") return `Under ${n(t.medium)}`;
  if (level === "Medium") return `${n(t.medium)} – ${n(t.high - 1)}`;
  if (level === "High") return `${n(t.high)} – ${n(t.critical - 1)}`;
  return `${n(t.critical)}+`;
}

/** Plain names for the two models. "Part A/B" is the dataset's name and means nothing to a visitor. */
export const MODEL_LABEL: Record<Part, { name: string; hint: string }> = {
  A: { name: "Dense crowds", hint: "packed scenes" },
  B: { name: "Sparse scenes", hint: "streets and open areas" },
};

/** A count the model cannot back to the last digit: two significant figures above 20. */
export function approxCount(n: number): string {
  const v = Math.max(0, Math.round(n));
  if (v < 20) return v.toLocaleString();
  const mag = 10 ** (String(v).length - 2);
  return (Math.round(v / mag) * mag).toLocaleString();
}


/** People-per-m² range for each level, from the API's band edges. */
export function densityRange(level: Level, b?: DensityBands): string {
  if (!b) return "—";
  if (level === "Low") return `Under ${b.medium}`;
  if (level === "Medium") return `${b.medium} – ${b.high}`;
  if (level === "High") return `${b.high} – ${b.critical}`;
  return `${b.critical}+`;
}

/** Rounded for display: one decimal under 10 people/m², whole numbers above. */
export function formatDensity(n: number): string {
  return n < 10 ? n.toFixed(1) : Math.round(n).toLocaleString();
}
