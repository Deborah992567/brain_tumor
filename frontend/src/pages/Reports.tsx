import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ReportListItem } from "../api/types";
import { Icon } from "../components/Icon";
import { EmptyState, ErrorState, SkeletonRows } from "../components/States";
import { useToast } from "../components/Toasts";
import { fmtBytes, fmtDateTime, fmtPercent } from "../utils/format";

const PAGE_SIZE = 10;

export function Reports() {
  const { toast } = useToast();
  const [items, setItems] = useState<ReportListItem[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api
      .reports({ page, page_size: PAGE_SIZE })
      .then((data) => {
        setItems(data.items);
        setTotal(data.total);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Could not load reports.");
      });
  }, [page]);

  useEffect(load, [load]);

  const download = async (item: ReportListItem) => {
    if (downloading) return;
    setDownloading(item.id);
    try {
      await api.downloadReport(item.download_url, `brain-mri-analysis-${item.analysis_id}.pdf`);
      toast("Report downloaded.", "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Download failed.", "error");
    } finally {
      setDownloading(null);
    }
  };

  if (error) return <ErrorState message={error} />;

  const lastPage = items ? Math.max(1, Math.ceil(total / PAGE_SIZE)) : 1;

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Reports</h1>
        <p className="page-subtitle">
          Professional PDF summaries of every analysis — patient section, interpretation,
          probability table and attention map.
        </p>
      </header>

      {!items ? (
        <SkeletonRows rows={4} />
      ) : items.length === 0 ? (
        <EmptyState
          title="No reports generated yet"
          hint="Analyze an image and click “Generate report” on the result page to create a printable PDF summary."
          action={
            <Link to="/analysis/new" className="btn btn-primary">
              <Icon name="plus" size={16} /> Analyze an image
            </Link>
          }
        />
      ) : (
        <div className="card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Prediction</th>
                  <th>Model probability</th>
                  <th>Model</th>
                  <th>Size</th>
                  <th>Generated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id}>
                    <td>{item.filename ?? item.analysis_id.slice(0, 8)}</td>
                    <td>
                      <span className="badge">{labelOf(item.prediction)}</span>
                    </td>
                    <td style={{ fontWeight: 600 }}>
                      {item.confidence != null ? fmtPercent(item.confidence) : "—"}
                    </td>
                    <td>
                      {item.model_name ?? "—"}{" "}
                      {item.model_version ? (
                        <span className="text-faint text-xs">v{item.model_version}</span>
                      ) : null}
                    </td>
                    <td className="text-muted">{fmtBytes(item.size_bytes)}</td>
                    <td className="text-muted" title={fmtDateTime(item.created_at)}>
                      {fmtDateTime(item.created_at)}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-sm"
                        disabled={downloading === item.id}
                        onClick={() => download(item)}
                      >
                        <Icon name="download" size={14} />
                        {downloading === item.id ? "Downloading…" : "Download"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex-between p-3">
            <span className="text-sm text-muted">
              {total} report{total === 1 ? "" : "s"}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                className="btn btn-sm"
                disabled={page <= 1}
                onClick={() => setPage(page - 1)}
              >
                Previous
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={page >= lastPage}
                onClick={() => setPage(page + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function labelOf(key: string | null): string {
  if (!key) return "—";
  const map: Record<string, string> = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    no_tumor: "No Tumor",
    pituitary_tumor: "Pituitary Tumor",
  };
  return map[key] ?? key;
}