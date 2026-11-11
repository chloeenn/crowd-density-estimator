import { useEffect, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  accept: string[];
  /** Upload size limit in MB, shown in the hint once the API reports it. */
  maxMb?: number;
  /** Slim single-line form, used once a photo is already on screen. */
  compact?: boolean;
  /** True once a photo is on screen (changes the wording only). */
  source?: boolean;
}

export function Dropzone({ onFile, disabled, accept, compact, source, maxMb }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file && !disabled) onFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile, disabled]);

  const open = () => !disabled && input.current?.click();

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-labelledby="dropzone-title"
      aria-describedby="dropzone-hint"
      onClick={open}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files[0];
        if (file && !disabled) onFile(file);
      }}
      className={`flex flex-col items-center justify-center gap-1.5 card card-lift px-6 text-center ${compact ? "py-5" : "py-10"} ${
        over ? "border-primary" : ""
      } ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
    >
      <p id="dropzone-title" className="font-display text-xl font-semibold text-primary">
        {source ? "Upload another photo" : "Upload a crowd photo"}
      </p>
      <p id="dropzone-hint" className="text-sm text-muted">
        Drop an image here, click to browse, or paste. JPEG, PNG or WebP{maxMb ? `, up to ${maxMb} MB` : ""}.
      </p>
      <input
        ref={input}
        type="file"
        accept={accept.join(",")}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />
    </div>
  );
}
