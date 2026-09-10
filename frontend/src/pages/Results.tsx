import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { PredictionResponse } from "../api/types";
import { AnalysisView } from "../components/AnalysisView";
import { ErrorState, PageLoader, SkeletonRows } from "../components/States";
import { useToast } from "../components/Toasts";

export function Results() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) {
      navigate("/history", { replace: true });
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .analysis(analysisId)
      .then((data) => {
        if (!cancelled) setResult(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load this analysis.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId, navigate]);

  const handleDownload = useCallback(
    (link: string, filename: string) => {
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
          window.setTimeout(() => URL.revokeObjectURL(url), 2500);
        })
        .catch(() => toast("Download failed.", "error"));
    },
    [toast],
  );

  if (error) {
    return (
      <ErrorState
        message={error}
        action={
          <button type="button" className="btn" onClick={() => navigate("/history")}>
            Back to history
          </button>
        }
      />
    );
  }

  if (loading) {
    return (
      <div>
        <SkeletonRows rows={3} />
        <PageLoader label="Loading analysis…" />
      </div>
    );
  }

  if (!result) return <PageLoader />;

  return (
    <div className="page" style={{ maxWidth: 920 }}>
      <header className="page-header flex-between" style={{ alignItems: "flex-end" }}>
        <div>
          <h1 className="page-title">Analysis Result</h1>
          <p className="page-subtitle">
            {result.filename ?? "Brain MRI"} — analyzed on{" "}
            {new Date(result.created_at).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "short",
            })}
          </p>
        </div>
        <button type="button" className="btn" onClick={() => navigate("/analysis/new")}>
          New analysis
        </button>
      </header>
      <AnalysisView result={result} onDownload={handleDownload} />
    </div>
  );
}