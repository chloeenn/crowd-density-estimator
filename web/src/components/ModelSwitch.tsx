import type { ModelDetail, Part } from "../api/types";

const HINT: Record<string, string> = { A: "dense crowds", B: "sparser street scenes" };

export function ModelSwitch({ models, value, onChange, disabled }: { models: ModelDetail[]; value: Part; onChange: (p: Part) => void; disabled?: boolean }) {
  if (models.length < 2) return null; // nothing to choose between
  return (
    <fieldset disabled={disabled} className="flex flex-wrap items-center gap-4">
      <legend className="sr-only">Model</legend>
      <span className="text-sm font-medium text-slate-900">Model</span>
      {models.map((m) => (
        <label key={m.part} className="flex items-center gap-1.5 text-sm text-slate-800">
          <input type="radio" name="model" checked={value === m.part} onChange={() => onChange(m.part)} />
          Part {m.part} <span className="text-slate-600">({HINT[m.part]})</span>
        </label>
      ))}
    </fieldset>
  );
}
