import type { ModelDetail, Part } from "../api/types";
import { MODEL_LABEL } from "../status";

export function ModelSwitch({ models, value, onChange, disabled }: { models: ModelDetail[]; value: Part; onChange: (p: Part) => void; disabled?: boolean }) {
  if (models.length < 2) return null; // nothing to choose between
  return (
    <fieldset disabled={disabled} className="space-y-3 disabled:opacity-50">
      <legend className="mb-3 font-display text-xl font-semibold text-primary">Scene type</legend>
      <div className="flex flex-wrap gap-3">
        {models.map((m) => (
          <label key={m.part} className="card card-lift flex min-h-11 cursor-pointer items-center gap-2 px-5 text-sm text-primary has-checked:border-primary">
            <input type="radio" name="model" className="accent-primary" checked={value === m.part} onChange={() => onChange(m.part)} />
            {MODEL_LABEL[m.part].name} <span className="text-muted">({MODEL_LABEL[m.part].hint})</span>
          </label>
        ))}
      </div>
      <p className="text-xs text-muted">Pick the type closest to your photo. The other model will give a poor estimate.</p>
    </fieldset>
  );
}
