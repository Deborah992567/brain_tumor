import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { DISCLAIMER_TEXT } from "../api/types";
import { Dropzone, type PendingFile } from "../components/Dropzone";
import { Icon } from "../components/Icon";
import { useToast } from "../components/Toasts";
import { fmtDuration } from "../utils/format";

type Phase = "pick" | "confirm" | "processing";

export function NewAnalysis() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [pending, setPending] = useState<PendingFile | null>(null);
  const [phase, setPhase] = useState<Phase>("pick");
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);

  const reset = () => {
    if (pending) URL.revokeObjectURL(pending.previewUrl);
    setPending(null);
    setPhase("pick");
    setProgress(0);
    setAnalyzing(false);
  };

  const onAccepted = (file: PendingFile) => {
    setPending(file);
    setPhase("confirm");
    setProgress(0);
  };

  const onDropError = (message: string) => toast(message, "error");

  const analyze = async () => {
    if (!pending || analyzing) return;
    setAnalyzing(true);
    setPhase("processing");
    setProgress(0);

    const started = performance.now();
    let result;
    try {
      result = await api.analyze(pending.file, setProgress);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed.";
      toast(message, "error");
      setAnalyzing(false);
      setProgress(0);
      setPhase("confirm");
      return;
    }

    const elapsed = performance.now() - started;
    const note = result.note ? ` ${result.note}` : "";
    toast(
      `Analysis complete — ${result.label} (model probability ${(result.confidence * 100).toFixed(1)}%). (${fmtDuration(
        elapsed,
      )})${result.low_confidence ? " Classification may be unavailable - check the result." : note}`,
      "success",
    );
    navigate(`/analysis/${result.analysis_id}`);
  };

  return (
    <div className="page" style={{ maxWidth: 860 }}>
      <header className="page-header">
        <h1 className="page-title">New Analysis</h1>
        <p className="page-subtitle">
          Upload a single brain MRI image. The active model produces a probability
          distribution across four classes with an adjustable trust threshold and a Grad-CAM
          attention overlay.
        </p>
      </header>

      {phase === "pick" && (
        <Dropzone
          onAccepted={onAccepted}
          onError={onDropError}
          disabled={false}
        />
      )}

      {phase === "confirm" && pending && (
        <div className="card">
          <h2 className="card-title mb-4">Confirm image</h2>
          <div className="preview-wrap">
            <img src={pending.previewUrl} alt="Selected MRI preview" className="preview-img" />
            <div>
              <p style={{ fontWeight: 600 }}>{pending.file.name}</p>
              <p className="text-sm text-muted">
                {pending.file.size / 1024 < 1024
                  ? `${(pending.file.size / 1024).toFixed(1)} KB`
                  : `${(pending.file.size / (1024 * 1024)).toFixed(2)} MB`}{" "}
                · {(pending.file.type || "image").replace(/^image\//, "").toUpperCase()} ·{" "}
                {pending.width}×{pending.height} px
              </p>
              <p className="text-sm text-muted">
                The image is resized to the model&apos;s expected input resolution before
                inference.
              </p>
              <div className="flex gap-3 mt-4">
                <button type="button" className="btn btn-primary" onClick={analyze} disabled={analyzing}>
                  <Icon name="upload" size={16} />
                  Analyze image
                </button>
                <button type="button" className="btn" onClick={reset} disabled={analyzing}>
                  Choose different
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {phase === "processing" && (
        <div className="card flex flex-center" style={{ flexDirection: "column", gap: 16, padding: 48 }}>
          <Icon name="history" size={36} className="text-faint" />
          <h2 className="card-title">Analyzing image…</h2>
          <p className="text-sm text-muted">Running model inference and generating attention map.</p>
          <div className="progress-track" style={{ width: "min(480px, 100%)" }}>
            <div
              className="progress-fill"
              style={{ width: `${Math.max(8, Math.min(100, progress * 100 + 8))}%` }}
            />
          </div>
          <p className="text-xs text-faint">
            {progress >= 1 ? "Processing…" : `Uploading ${Math.round(progress * 100)}%`}
          </p>
        </div>
      )}

      <section className="section mt-6">
        <div className="disclaimer">
          <strong>Disclaimer:</strong> {DISCLAIMER_TEXT}
        </div>
      </section>
    </div>
  );
}