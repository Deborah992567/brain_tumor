import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { ModelInfo, ModelListResponse } from "../api/types";
import { Icon } from "../components/Icon";
import { Modal } from "../components/Modal";
import { EmptyState, ErrorState, SkeletonRows } from "../components/States";
import { useToast } from "../components/Toasts";
import { fmtDateTime, fmtPercent } from "../utils/format";

const HONEST_HINT =
  "Metrics were recorded at training time and have not been re-validated against a held-out clinical set. Treat them as indicative only.";

export function Models() {
  const { toast } = useToast();
  const [data, setData] = useState<ModelListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<ModelInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [pendingActivate, setPendingActivate] = useState<ModelInfo | null>(null);

  const load = useCallback(() => {
    setError(null);
    api
      .models()
      .then(setData)
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Could not load models.");
      });
  }, []);

  useEffect(load, [load]);

  const activate = async (model: ModelInfo) => {
    setBusy(true);
    try {
      await api.activateModel(model.id);
      toast(`“${model.name}” is now the active model.`, "success");
    } catch (err) {
      toast(
        err instanceof Error ? err.message : "Activation failed. Check the admin key.",
        "error",
      );
    } finally {
      setBusy(false);
      setPendingActivate(null);
      load();
    }
  };

  const reload = async () => {
    setBusy(true);
    try {
      const res = await api.reloadModels();
      toast(res.status === "ok" ? "Models refreshed from disk." : "Model scan returned an unexpected status.", "info");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Reload failed.", "error");
    } finally {
      setBusy(false);
      load();
    }
  };

  if (error) return <ErrorState message={error} />;

  return (
    <div>
      <header className="page-header">
        <div className="flex-between" style={{ alignItems: "flex-end" }}>
          <div>
            <h1 className="page-title">Models</h1>
            <p className="page-subtitle">
              Inspect registered models, their honest training-time metrics and switch the
              active inference model.
            </p>
          </div>
          <button type="button" className="btn" onClick={reload} disabled={busy}>
            <Icon name="settings" size={15} /> {busy ? "Scanning…" : "Rescan models"}
          </button>
        </div>
      </header>

      {!data ? (
        <SkeletonRows rows={3} />
      ) : data.models.length === 0 ? (
        <EmptyState
          title="No models registered"
          hint="The registry at backend/models/registry.json is empty or unreachable. Place a trained model file (.h5) next to it and rescan."
        />
      ) : (
        <div className="card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Architecture</th>
                  <th>Status</th>
                  <th>Training accuracy</th>
                  <th>Validation accuracy</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.models.map((model) => {
                  const trainAcc = metricPct(model, "train_accuracy");
                  const valAcc = metricPct(model, "val_accuracy");
                  return (
                    <tr key={model.id} className={model.is_active ? "model-active" : undefined}>
                      <td>
                        <div style={{ fontWeight: 650 }}>
                          {model.name}
                          {model.is_active ? (
                            <span className="active-tag"> · active</span>
                          ) : null}
                        </div>
                        <div className="text-xs text-faint">v{model.version}</div>
                      </td>
                      <td className="text-sm">
                        {model.architecture} · {model.input_size}px
                      </td>
                      <td>
                        <span className="badge badge-success">{model.status}</span>
                      </td>
                      <td>{trainAcc != null ? fmtPercent(trainAcc) : "—"}</td>
                      <td>{valAcc != null ? fmtPercent(valAcc) : "—"}</td>
                      <td>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            className="btn btn-sm"
                            onClick={() => setDetail(model)}
                          >
                            Details
                          </button>
                          {!model.is_active && (
                            <button
                              type="button"
                              className="btn btn-primary btn-sm"
                              onClick={() => setPendingActivate(model)}
                              disabled={busy}
                            >
                              Activate
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <section className="section">
        <div className="disclaimer">{HONEST_HINT}</div>
      </section>

      {data?.active && <ActiveModelSummary model={data.active} />}

      {detail && <ModelDetailModal model={detail} onClose={() => setDetail(null)} />}

      {pendingActivate && (
        <Modal
          title="Activate model"
          onClose={() => setPendingActivate(null)}
          footer={
            <>
              <button type="button" className="btn" onClick={() => setPendingActivate(null)}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => activate(pendingActivate)}
                disabled={busy}
              >
                Confirm activation
              </button>
            </>
          }
        >
          <p className="text-sm">
            New analyses will use <strong>{pendingActivate.name} v{pendingActivate.version}</strong>.
          </p>
        </Modal>
      )}
    </div>
  );
}

function ActiveModelSummary({ model }: { model: ModelInfo }) {
  const metrics = useMemo(
    () =>
      model.metrics && typeof model.metrics === "object"
        ? Object.entries(model.metrics).filter(
            ([key, value]) => typeof value === "number" && key !== "num_classes",
          )
        : [],
    [model.metrics],
  );

  return (
    <section className="card section">
      <h2 className="card-title mb-3">Active model details</h2>
      <div className="grid grid-2">
        <Metric label="Name" value={`${model.name} v${model.version}`} />
        <Metric label="Architecture" value={model.architecture} />
        <Metric label="Input size" value={`${model.input_size}×${model.input_size} px`} />
        <Metric label="Dataset version" value={model.dataset_version || "—"} />
        <Metric label="Registered" value={fmtDateTime(model.created_at)} />
        <Metric label="Status" value={model.status} />
      </div>
      {model.description && <p className="text-sm text-muted mt-3">{model.description}</p>}
      {metrics.length > 0 && (
        <div className="card section" style={{ marginTop: 16 }}>
          <h3 className="section-title text-sm">Training-time metrics</h3>
          <div className="grid grid-2">
            {metrics.map(([key, value]) => (
              <Metric key={key} label={key} value={fmtPercent(value as number)} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ModelDetailModal({ model, onClose }: { model: ModelInfo; onClose: () => void }) {
  const metrics = useMemo(
    () =>
      model.metrics && typeof model.metrics === "object"
        ? Object.entries(model.metrics)
        : [],
    [model.metrics],
  );

  return (
    <Modal title={`${model.name} v${model.version}`} onClose={onClose}>
      <div className="grid grid-2">
        <Metric label="Architecture" value={model.architecture} />
        <Metric label="Input size" value={`${model.input_size}px`} />
        <Metric label="Dataset" value={model.dataset_version || "—"} />
        <Metric label="Status" value={model.status} />
        <Metric label="File" value={model.file_path} />
        <Metric label="Registered" value={fmtDateTime(model.created_at)} />
      </div>
      {model.description && <p className="text-sm text-muted mt-3">{model.description}</p>}
      {metrics.length > 0 && (
        <div className="mt-4">
          <h3 className="section-title text-sm">Metrics</h3>
          <div className="grid grid-2">
            {metrics.map(([key, value]) => (
              <Metric
                key={key}
                label={key}
                value={typeof value === "number" ? fmtPercent(value) : String(value)}
              />
            ))}
          </div>
          <p className="text-xs text-faint mt-3">{HONEST_HINT}</p>
        </div>
      )}
    </Modal>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-faint">{label}</div>
      <div style={{ fontWeight: 550, fontSize: 14, wordBreak: "break-word" }}>{value}</div>
    </div>
  );
}

function metricPct(model: ModelInfo, key: string): number | null {
  const value = model.metrics?.[key];
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}