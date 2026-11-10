import type { PredictResponse } from "../api/types";
import { LevelBadge } from "./LevelBadge";

export function ResultCard({ result, caveat }: { result: PredictResponse; caveat?: string }) {
  return (
    <section aria-label="Result" className="space-y-3 rounded-xl border border-slate-300 bg-white p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-4xl font-bold tabular-nums text-slate-900">
          {Math.round(result.count).toLocaleString()}
          <span className="ml-2 text-base font-normal text-slate-600">people (estimated)</span>
        </p>
        <LevelBadge level={result.level} />
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm text-slate-700">
        <div>
          <dt className="font-medium text-slate-900">Local concentration</dt>
          <dd className="tabular-nums">{result.local_concentration.toFixed(2)} per cell</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-900">Inference time</dt>
          <dd className="tabular-nums">{(result.inference_ms / 1000).toFixed(1)} s</dd>
        </div>
        <div>
          <dt className="font-medium text-slate-900">Model</dt>
          <dd>
            {result.model.version} (Part {result.model.part})
          </dd>
        </div>
      </dl>
      {caveat && <p className="border-t border-slate-200 pt-3 text-sm text-slate-600">{caveat}</p>}
    </section>
  );
}
