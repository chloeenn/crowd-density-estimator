import { useState, type ReactNode } from "react";
import type { ModelDetail, Scene } from "../api/types";

const SEGMENTS: Record<Scene, { name: string; hint: string }> = {
  auto: { name: "Automatic", hint: "Picks the model that fits each photo." },
  A: { name: "Dense", hint: "Dense crowds: for packed scenes." },
  B: { name: "Sparse", hint: "Sparse scenes: for streets and open areas." },
};

/** Scene type as a three-way switch, tucked behind one pill: Automatic is right for almost every photo, so it stays
 * out of the way. `actions` sit in the same row as the pill so the buttons read as one tidy group. */
export function ModelSwitch({ models, value, onChange, disabled, auto, actions }: { models: ModelDetail[]; value: Scene; onChange: (s: Scene) => void; disabled?: boolean; auto: boolean; actions?: ReactNode }) {
  const [open, setOpen] = useState(false);
  if (models.length < 2) return actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null; // nothing to choose between
  const ids: Scene[] = [...(auto ? (["auto"] as const) : []), ...models.map((m) => m.part as Scene)];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <button type="button" className="pill pill-ghost gap-2" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          Scene type: {SEGMENTS[value].name}
          <svg aria-hidden="true" viewBox="0 0 12 12" className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2.5 4.5 6 8l3.5-3.5" />
          </svg>
        </button>
        {actions}
      </div>
      <fieldset hidden={!open} disabled={disabled} className="space-y-2 disabled:opacity-50">
        <legend className="sr-only">Scene type</legend>
        <div className="inline-flex max-w-full flex-wrap gap-1 rounded-full bg-soft p-1">
          {ids.map((id) => (
            <label
              key={id}
              className="flex min-h-10 cursor-pointer items-center rounded-full px-4 text-sm text-primary transition-colors has-checked:bg-primary has-checked:text-page has-focus-visible:outline-2 has-focus-visible:outline-primary"
            >
              <input type="radio" name="model" className="sr-only" checked={value === id} onChange={() => onChange(id)} />
              {SEGMENTS[id].name}
            </label>
          ))}
        </div>
        <p className="text-sm text-muted">{SEGMENTS[value].hint}</p>
      </fieldset>
    </div>
  );
}
