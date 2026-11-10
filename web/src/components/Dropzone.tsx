import { useEffect, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
  accept: string[];
}

export function Dropzone({ onFile, disabled, accept }: Props) {
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
      aria-label="Upload a crowd photo. Drop an image here, press Enter to browse, or paste from the clipboard."
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
      className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed p-8 text-center transition focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-400 ${
        over ? "border-indigo-600 bg-indigo-50" : "border-slate-400 bg-white hover:bg-slate-50"
      } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
    >
      <p className="text-base font-medium text-slate-900">Drop a crowd photo here, or click to browse</p>
      <p className="text-sm text-slate-600">You can also paste an image (Ctrl/Cmd+V). JPEG, PNG or WebP.</p>
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
