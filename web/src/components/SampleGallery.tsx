import type { Part } from "../api/types";

export interface Sample {
  src: string;
  label: string;
  /** The model this scene was picked for. Choosing the sample selects it. */
  part: Part;
}

export const SAMPLES: Sample[] = [
  { src: "/samples/part-a-1.jpg", label: "Dense crowd 1", part: "A" },
  { src: "/samples/part-a-2.jpg", label: "Dense crowd 2", part: "A" },
  { src: "/samples/part-b-1.jpg", label: "Street scene 1", part: "B" },
  { src: "/samples/part-b-2.jpg", label: "Street scene 2", part: "B" },
];

export function SampleGallery({ onPick, disabled }: { onPick: (s: Sample) => void; disabled?: boolean }) {
  return (
    <section aria-label="Sample images" className="space-y-3">
      <h2 className="text-2xl md:text-3xl">Or try a sample</h2>
      <p className="text-sm text-muted">Tap a photo to see a result right away.</p>
      <ul className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {SAMPLES.map((s) => (
          <li key={s.src}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(s)}
              className="card card-lift block w-full overflow-hidden text-left text-sm font-medium text-primary disabled:opacity-40"
            >
              <img src={s.src} alt="" className="aspect-4/3 w-full object-cover" loading="lazy" />
              <span className="block px-4 py-2">{s.label}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
