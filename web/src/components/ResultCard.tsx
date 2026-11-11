import type { Ref } from "react";
import type { PredictResponse } from "../api/types";
import { Pips } from "./Pips";
import { MODEL_LABEL, approxCount, levelStyle } from "../status";

export function ResultCard({ result, headingRef, stale, partLabel }: { result: PredictResponse; headingRef?: Ref<HTMLHeadingElement>; stale?: boolean; partLabel?: string }) {
  const part = result.model.part === "B" ? "B" : "A";
  const level = levelStyle(result.level);
  const count = approxCount(result.count);
  return (
    <section aria-labelledby="result-heading" className="space-y-5">
      <div className="space-y-4">
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
          aria-label={`Estimate: about ${count} people, ${level.label}`}
          className="font-display text-xl font-semibold text-primary"
        >
          Estimate
        </h2>
        <p aria-hidden="true" className="flex items-baseline gap-3">
          <span className="font-display text-6xl font-semibold tracking-tight tabular-nums text-primary" style={{ fontFeatureSettings: '"lnum"' }}>
            ~{count}
          </span>
          <span className="text-base text-muted">people</span>
        </p>
        <div className="space-y-1.5">
          <p className="flex items-center gap-3 text-xl font-medium text-primary">
            <Pips level={result.level} />
            {level.label}
          </p>
        </div>
      </div>
      <dl className="space-y-3 pt-2 text-sm">
        <div className="flex justify-between gap-4">
          <dt className="text-muted">Scene type</dt>
          <dd className="text-right text-primary">{MODEL_LABEL[part].name}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted">Analysis time</dt>
          <dd className="text-right tabular-nums text-primary">{(result.inference_ms / 1000).toFixed(1)} s</dd>
        </div>
      </dl>
    </section>
  );
}
