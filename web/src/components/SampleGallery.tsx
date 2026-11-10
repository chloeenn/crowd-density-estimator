export const SAMPLES = [
  { src: "/samples/part-a-1.jpg", label: "Dense crowd (sample 1)" },
  { src: "/samples/part-a-2.jpg", label: "Dense crowd (sample 2)" },
  { src: "/samples/part-b-1.jpg", label: "Street scene (sample 1)" },
  { src: "/samples/part-b-2.jpg", label: "Street scene (sample 2)" },
];

export function SampleGallery({ onPick, disabled }: { onPick: (src: string, label: string) => void; disabled?: boolean }) {
  return (
    <section aria-label="Sample images" className="space-y-2">
      <h2 className="text-sm font-semibold text-slate-900">Or try a sample</h2>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {SAMPLES.map((s) => (
          <li key={s.src}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onPick(s.src, s.label)}
              className="block w-full overflow-hidden rounded-lg border border-slate-300 focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-400 disabled:opacity-60"
            >
              <img src={s.src} alt={s.label} className="aspect-4/3 w-full object-cover" loading="lazy" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
