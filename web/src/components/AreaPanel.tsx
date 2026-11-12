import { useState } from "react";
import type { Point } from "../api/types";

export interface AreaValue {
  areaM2?: number;
  region: Point[];
}

interface Props {
  value: AreaValue;
  applied: boolean;
  drawing: boolean;
  disabled?: boolean;
  onDrawing: (on: boolean) => void;
  onRegion: (r: Point[]) => void;
  onApply: (v: AreaValue) => void;
  onRemove: () => void;
}

type Scope = "whole" | "part";

/** Turns the head count into people per m², how tightly packed the crowd is. A photo cannot show the real size of the
 * ground, so the user supplies it: for the whole photo, or for an outlined part of it. */
export function AreaPanel({ value, applied, drawing, disabled, onDrawing, onRegion, onApply, onRemove }: Props) {
  const [scope, setScope] = useState<Scope>(value.region.length > 0 ? "part" : "whole");
  const [text, setText] = useState(value.areaM2 ? String(value.areaM2) : "");
  const area = Number(text);
  const validArea = text.trim() !== "" && Number.isFinite(area) && area > 0;
  const hasOutline = value.region.length >= 3;
  const canApply = validArea && !drawing && (scope === "whole" || hasOutline);

  const choose = (s: Scope) => {
    setScope(s);
    onDrawing(false);
    if (s === "whole") onRegion([]);
  };

  return (
    <section aria-labelledby="area-heading" className="card grid gap-6 p-6 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] md:gap-10">
      <div className="space-y-3">
        <h2 id="area-heading" className="font-display text-xl font-semibold text-primary">How packed is it?</h2>
        <p className="text-sm text-muted">
          A head count cannot say how crowded a place feels. Enter the real size of the ground to get <strong className="font-medium text-primary">people per m²</strong> and a
          level from Low to Critical.
        </p>
        <fieldset disabled={disabled} className="space-y-2">
          <legend className="sr-only">Your area is</legend>
          <div className="inline-flex max-w-full flex-wrap gap-1 rounded-full bg-soft p-1">
            {(
              [
                ["whole", "The whole photo"],
                ["part", "Part of the photo"],
              ] as const
            ).map(([id, label]) => (
              <label
                key={id}
                className="flex min-h-10 cursor-pointer items-center rounded-full px-4 text-sm text-primary transition-colors has-checked:bg-primary has-checked:text-page has-focus-visible:outline-2 has-focus-visible:outline-primary"
              >
                <input type="radio" name="area-scope" className="sr-only" checked={scope === id} onChange={() => choose(id)} />
                {label}
              </label>
            ))}
          </div>
          <p className="text-sm text-muted">
            {scope === "whole" ? "You know how much ground the photo shows." : "You know the size of one area in it, such as a plaza or a queue. Only the people inside the outline are counted."}
          </p>
        </fieldset>
      </div>

      <div className="space-y-3">
        {scope === "part" && (
          <div className="space-y-2 rounded-2xl bg-weight/50 p-3 text-sm">
            {drawing ? (
              <>
                <p role="status" className="text-muted">
                  Click the photo above to place each corner of the area. {value.region.length} placed, at least 3 needed.
                </p>
                <div className="flex flex-wrap gap-2">
                  <button type="button" className="pill pill-solid disabled:opacity-40" disabled={value.region.length < 3} onClick={() => onDrawing(false)}>
                    Done outlining
                  </button>
                  <button type="button" className="pill pill-ghost" onClick={() => onRegion(value.region.slice(0, -1))} disabled={value.region.length === 0}>
                    Undo last corner
                  </button>
                </div>
              </>
            ) : (
              <div className="flex flex-wrap items-center gap-3">
                <button type="button" className="pill pill-ghost" disabled={disabled} onClick={() => (onRegion([]), onDrawing(true))}>
                  {hasOutline ? "Outline again" : "Outline the area on the photo"}
                </button>
                {hasOutline && <span className="text-muted">Area outlined.</span>}
              </div>
            )}
          </div>
        )}

        <form
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (canApply) onApply({ areaM2: area, region: scope === "part" ? value.region : [] });
          }}
        >
          <label className="block space-y-1 text-sm">
            <span className="text-primary">{scope === "part" ? "Size of the outlined area (m²)" : "Ground area shown in the photo (m²)"}</span>
            <span className="flex flex-wrap items-center gap-3">
              <input
                inputMode="decimal"
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={disabled}
                aria-invalid={text.trim() !== "" && !validArea}
                placeholder="e.g. 200"
                className="min-h-11 w-32 rounded-full border border-edge bg-surface px-4 tabular-nums"
              />
              <span className="text-muted">e.g. 10 m × 20 m = 200 m²</span>
            </span>
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="submit" className="pill pill-solid disabled:opacity-40" disabled={disabled || !canApply}>
              Get people per m²
            </button>
            {applied && (
              <button type="button" className="pill pill-ghost" disabled={disabled} onClick={() => (setText(""), setScope("whole"), onRemove())}>
                Remove
              </button>
            )}
          </div>
          {scope === "part" && !hasOutline && !drawing && <p className="text-sm text-muted">Outline the area first, then enter its size.</p>}
        </form>
      </div>
    </section>
  );
}
