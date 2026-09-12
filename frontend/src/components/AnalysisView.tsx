import { useState } from "react";
import { api } from "../api/client";
import type { PredictionResponse } from "../api/types";
import { CLASS_LABELS, CLASS_ORDER, DISCLAIMER_TEXT } from "../api/types";
import { fmtDateTime, fmtDuration, fmtPercent } from "../utils/format";
import { Modal } from "./Modal";
import { useToast } from "./Toasts";

export function AnalysisView({
  result,
  onDownload,
}: {
  result: PredictionResponse;
  onDownload?: (link: string, filename: string) => void;
}) {
  const toast = useToast();
  const [reportLoading, setReportLoading] = useState(false);
  const [imageModal, setImageModal] = useState<string | null>(null);

  const download = (link: string | null, filename: string) => {
    if (!link) return;
    void fetch(link)
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2500);
      });
  };

  const handleDownload = (link: string, filename: string) => {
    if (onDownload) {
      onDownload(link, filename);
    } else {
      download(link, filename);
    }
  };

  const generateReport = async () => {
    setReportLoading(true);
    try {
      const report = await api.generateReport(result.analysis_id);
      toast.toast(`Report generated (${fmtBytes(report.size_bytes)}).`, "success");
      if (report.download_url) {
        handleDownload(report.download_url, `brain-tumor-analysis-${result.analysis_id}.pdf`);
      }
    } catch (err) {
      toast.toast(err instanceof Error ? err.message : "Could not generate report.", "error");
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <div>
      <div className="media-compare">
        {result.links.image && (
          <figure className="media-figure">
            <img
              src={result.links.image}
              alt="Submitted MRI"
              onClick={() => setImageModal(result.links.image)}
              style={{ cursor: "zoom-in" }}
            />
            <figcaption className="media-caption">Submitted image</figcaption>
          </figure>
        )}
        {result.links.gradcam && (
          <figure className="media-figure">
            <img
              src={result.links.gradcam}
              alt="AI attention visualization overlay"
              onClick={() => setImageModal(result.links.gradcam)}
              style={{ cursor: "zoom-in" }}
            />
            <figcaption className="media-caption">AI attention visualization</figcaption>
          </figure>
        )}
      </div>

      <section className="card section" aria-label="Prediction result">
        <div className="flex-between mb-3">
          <h2 className="card-title">Prediction</h2>
        </div>

        {result.low_confidence && (
          <div className="alert alert-warning mb-3" role="note">
            <strong>Classification unavailable.</strong> The model probability is
            below the reliability threshold, so this result should not be used to
            draw a conclusion. Treat the class below as indicative only.
          </div>
        )}

        <div className="prediction-box" style={result.low_confidence ? { borderLeftColor: "var(--warning)" } : undefined}>
          <div className="prediction-label">
            {result.low_confidence ? "Uncertain result" : result.label}
          </div>
          <div className="confidence-value">
            Model probability {fmtPercent(result.confidence)}
          </div>
        </div>

        <div className="alert alert-info mt-3" role="note">
          {result.supported_classes && result.supported_classes.length > 0 ? (
            <>
              This model recognizes exactly:{" "}
              {result.supported_classes.map((key: string) => CLASS_LABELS[key as keyof typeof CLASS_LABELS] ?? key).join(", ")}.
              {" "}Other tumor types are outside its label space, and results for
              such images are unreliable even when the probability is high.
            </>
          ) : (
            <>
              This model recognizes exactly:{" "}
              {CLASS_ORDER.map((key) => CLASS_LABELS[key]).join(", ")}. Other tumor
              types are outside its label space, and results for such images are
              unreliable even when the probability is high.
            </>
          )}
        </div>

        <div className="mt-5">
          <h3 className="section-title text-sm">Class probabilities</h3>
          {CLASS_ORDER.map((key, index) => {
            const prob = result.probabilities[key] ?? 0;
            return (
              <div className="prob-row" key={key}>
                <div className="prob-label">{CLASS_LABELS[key]}</div>
                <div className="prob-track">
                  <div
                    className={`prob-fill${index === 0 ? " top" : ""}`}
                    style={{ width: `${Math.max(0, Math.min(100, prob * 100))}%` }}
                  />
                </div>
                <div className="prob-value">{fmtPercent(prob)}</div>
              </div>
            );
          })}
        </div>

        <div className="grid grid-2" style={{ marginTop: 24 }}>
          <InfoRow label="Analysis ID" value={result.analysis_id} />
          <InfoRow label="Analyzed at" value={fmtDateTime(result.created_at)} />
          <InfoRow
            label="Model"
            value={`${result.model.name} v${result.model.version}`}
          />
          <InfoRow label="Inference time" value={fmtDuration(result.processing_time_ms)} />
        </div>

        {result.note && (
          <div className="alert alert-warning mt-5" role="note">
            {result.note}
          </div>
        )}

        <div className="flex gap-3 mt-5" style={{ flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={generateReport}
            disabled={reportLoading}
          >
            {reportLoading ? "Generating…" : "Generate report"}
          </button>
          {result.links.gradcam && (
            <button
              type="button"
              className="btn"
              onClick={() => handleDownload(result.links.gradcam as string, "attention-overlay.png")}
            >
              Download overlay
            </button>
          )}
        </div>
      </section>

      <div className="disclaimer">
        <strong>Disclaimer:</strong> {DISCLAIMER_TEXT}
      </div>

      {imageModal && (
        <Modal title="Enlarged image" onClose={() => setImageModal(null)}>
          <img src={imageModal} alt="Enlarged view" style={{ width: "100%", borderRadius: 8 }} />
        </Modal>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-faint">{label}</div>
      <div style={{ fontWeight: 550, fontSize: 14 }}>{value}</div>
    </div>
  );
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}