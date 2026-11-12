import type { Ref } from "react";
import type { PredictResponse } from "../api/types";
import { Pips } from "./Pips";
import { MODEL_LABEL, SIZES, approxCount, formatDensity, levelStyle } from "../status";

export function ResultCard({ result, headingRef, stale, partLabel }: { result: PredictResponse; headingRef?: Ref<HTMLHeadingElement>; stale?: boolean; partLabel?: string }) {
  const part = result.model.part === "B" ? "B" : "A";
  const d = result.density;
  const level = d ? levelStyle(d.level) : { ...levelStyle(result.level), label: SIZES[result.level].label };
  const shownLevel = d ? d.level : result.level;
  const count = approxCount(d ? d.count : result.count);
  const where = d?.in_region ? "in the marked area" : "in the photo";
  return (
    <section aria-labelledby="result-heading" className="space-y-3">
      <div className="space-y-2">
        {stale && (
          <p role="status" className="inline-flex items-center gap-2 rounded-full border border-edge bg-surface px-3 py-1 text-sm text-primary">
            <span aria-hidden="true" className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            Re-running{partLabel ? ` on ${partLabel}` : ""}… this is the previous result.
          </p>
        )}
        {/* The label names the whole verdict, so focus landing here reads the number, not just "Estimate". */}
        <h2
          id="result-heading"
          ref={headingRef}
          tabIndex={-1}
          aria-label={d ? `Estimate: ${formatDensity(d.people_per_m2)} people per square metre, ${level.label}` : `Estimate: about ${count} people, ${level.label}`}
          className="font-display text-xl font-semibold text-primary"
        >
          Estimate
        </h2>
        <p aria-hidden="true" className="flex items-baseline gap-3">
          <span className="font-display text-6xl font-semibold tracking-tight tabular-nums text-primary" style={{ fontFeatureSettings: '"lnum"' }}>
            {d ? formatDensity(d.people_per_m2) : `~${count}`}
          </span>
          <span className="text-base text-muted">{d ? "people per m²" : "people"}</span>
        </p>
        {d && (
          <p aria-hidden="true" className="text-sm text-muted">
            About {count} people {where}, on {d.area_m2.toLocaleString()} m².
          </p>
        )}
        <div className="space-y-1.5">
          <p className="flex items-center gap-3 text-xl font-medium text-primary">
            <Pips level={shownLevel} />
            {level.label}
          </p>
        </div>
      </div>
      <dl className="space-y-1.5 text-sm">
        <div className="flex justify-between gap-4">
          <dt className="text-muted">Scene type</dt>
          <dd className="text-right text-primary">
            {MODEL_LABEL[part].name}
            {result.routed ? " (automatic)" : ""}
          </dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted">Analysis time</dt>
          <dd className="text-right tabular-nums text-primary">{(result.inference_ms / 1000).toFixed(1)} s</dd>
        </div>
      </dl>
    </section>
  );
}
