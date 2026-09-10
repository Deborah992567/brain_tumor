import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { HealthResponse, HistoryItem, ModelInfo } from "../api/types";
import { Icon } from "../components/Icon";
import { EmptyState, ErrorState, SkeletonRows } from "../components/States";
import { fmtPercent } from "../utils/format";
import { DISCLAIMER_TEXT } from "../api/types";

export function Dashboard() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryItem[] | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([api.history({ page: 1, page_size: 6 }), api.models(), api.health()])
      .then(([h, m, hh]) => {
        if (cancelled) return;
        if (h.status === "fulfilled") setHistory(h.value.items);
        if (m.status === "fulfilled") setModel(m.value.active);
        if (hh.status === "fulfilled") setHealth(hh.value);
      })
      .catch(() => {
        if (!cancelled) setError("Could not load dashboard data.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <ErrorState message={error} />;

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Monitor analyses, the active model and service health in one place.
        </p>
      </header>

      {/* Quick actions */}
      <div className="flex gap-3 mb-5">
        <button type="button" className="btn btn-primary btn-lg" onClick={() => navigate("/analysis/new")}>
          <Icon name="upload" size={16} />
          Analyze MRI
        </button>
        <button type="button" className="btn btn-lg" onClick={() => navigate("/history")}>
          View History
        </button>
      </div>

      {!history && !error ? (
        <SkeletonRows rows={3} />
      ) : history && history.length === 0 ? (
        <EmptyState
          title="No analyses yet"
          hint="Upload a brain MRI image to run an AI-assisted analysis. Predictions are stored in your history for later review."
          action={
            <button type="button" className="btn btn-primary" onClick={() => navigate("/analysis/new")}>
              <Icon name="upload" size={16} /> Upload first image
            </button>
          }
        />
      ) : (
        <>
          {/* Stat cards */}
          <section className="grid grid-4 section">
            <StatCard label="Total analyses" value={historyCount(history)} />
            <StatCard label="Active model" value={modelName(model)} />
            <StatCard label="Model version" value={modelVersion(model)} />
            <StatCard label="Service" value={health?.status === "ok" ? "Online" : "Offline"} tone={health?.status === "ok" ? "success" : "danger"} />
          </section>

          <div className="grid grid-2 section">
            {/* Recent analyses */}
            <section className="card">
              <div className="flex-between mb-3">
                <h2 className="card-title">Recent analyses</h2>
                <Link to="/history" className="text-sm">
                  View all
                </Link>
              </div>
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {(history ?? []).slice(0, 5).map((item) => (
                  <li key={item.id}>
                    <Link
                      to={`/analysis/${item.id}`}
                      className="flex-between"
                      style={{ padding: "10px 0", borderBottom: "1px solid var(--border)" }}
                    >
                      <div>
                        <div style={{ fontWeight: 600 }}>{labelOf(item.prediction)}</div>
                        <div className="text-xs text-faint">{formatDate(item.created_at)}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm" style={{ fontWeight: 600 }}>
                          {fmtPercent(item.confidence)}
                        </div>
                        <div className="text-xs text-faint">confidence</div>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>

            {/* Active model */}
            <section className="card">
              <h2 className="card-title mb-3">Active model</h2>
              {model ? (
                <div>
                  <div className="text-lg" style={{ fontWeight: 650 }}>
                    {model.name}
                  </div>
                  <div className="text-sm text-muted mt-2">
                    v{model.version} — {model.architecture}
                  </div>
                  <div className="text-sm text-muted mt-2">
                    Input size {model.input_size}px · {model.dataset_version}
                  </div>
                  <Link to="/models" className="btn btn-sm mt-4">
                    Manage models
                  </Link>
                </div>
              ) : (
                <p className="text-muted text-sm">No active model configured.</p>
              )}
            </section>
          </div>
        </>
      )}

      <section className="section">
        <div className="disclaimer">
          <strong>Medical disclaimer:</strong> {DISCLAIMER_TEXT}
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "success" | "danger";
}) {
  const color =
    tone === "success" ? "var(--success)" : tone === "danger" ? "var(--danger)" : undefined;
  return (
    <div className="card stat">
      <div className="stat-value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function historyCount(history: HistoryItem[] | null): string {
  return history ? String(history.length) : "—";
}

function modelName(model: ModelInfo | null): string {
  return model ? model.name : "—";
}

function modelVersion(model: ModelInfo | null): string {
  return model ? model.version : "—";
}

function labelOf(key: string): string {
  const map: Record<string, string> = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    no_tumor: "No Tumor",
    pituitary_tumor: "Pituitary Tumor",
  };
  return map[key] ?? key;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}