import { useCallback, useRef, useState } from "react";
import { Icon } from "./Icon";

const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"];
const MAX_SIZE_MB = 10;

export interface PendingFile {
  file: File;
  previewUrl: string;
  width: number;
  height: number;
}

interface DropzoneProps {
  onAccepted: (pending: PendingFile) => void;
  onError: (message: string) => void;
  disabled?: boolean;
}

export function validateImageFile(file: File): string | null {
  const ext = file.name.includes(".")
    ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
    : "";
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    return "Unsupported format. Use JPG, PNG, BMP, TIFF or JPEG.";
  }
  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > MAX_SIZE_MB) {
    return `File exceeds the ${MAX_SIZE_MB} MB upload limit.`;
  }
  return null;
}

async function readDimensions(file: File): Promise<{ width: number; height: number }> {
  const url = URL.createObjectURL(file);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => resolve({ width: 0, height: 0 });
    img.src = url;
  });
}

export function Dropzone({ onAccepted, onError, disabled }: DropzoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      const error = validateImageFile(file);
      if (error) {
        onError(error);
        return;
      }
      const dims = await readDimensions(file);
      if (dims.width < 64 || dims.height < 64) {
        onError("Image is too small. Minimum 64x64 pixels is required.");
        return;
      }
      onAccepted({
        file,
        previewUrl: URL.createObjectURL(file),
        width: dims.width,
        height: dims.height,
      });
    },
    [onAccepted, onError],
  );

  return (
    <div>
      <div
        className={`dropzone${dragOver ? " drag-over" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a brain MRI image"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragOver(false);
          if (disabled) return;
          const file = event.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
      >
        <div className="dropzone-icon">
          <Icon name="upload" size={40} />
        </div>
        <div className="dropzone-title">Drag and drop a brain MRI image here</div>
        <p className="dropzone-hint">or click to browse your computer</p>
        <div className="dropzone-formats">
          {ACCEPTED_EXTENSIONS.map((ext) => (
            <span key={ext} className="format-chip">
              {ext}
            </span>
          ))}
        </div>
        <p className="dropzone-hint mt-3">
          Max {MAX_SIZE_MB} MB — a single axial, coronal or sagittal brain scan.
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) handleFile(file);
          event.target.value = "";
        }}
      />
    </div>
  );
}