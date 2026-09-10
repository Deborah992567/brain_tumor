import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { HistoryItem } from "../api/types";
import { Icon } from "../components/Icon";
import { EmptyState, ErrorState, SkeletonRows } from "../components/States";
import { fmtDateTime, fmtPercent } from "../utils/format";

const PAGE_SIZE = 10;

export function History() {
  const navigate = useNavigate();
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search), 350);
    return () => window.clearTimeout(timer);
  }, [search]);

  const load = useCallback(() => {
    setError(null);
    api
      .history({
        page,
        page_size: PAGE_SIZE,
        search: debouncedSearch || undefined,
        prediction: prediction || undefined,
      })
      .then((data) => {
        setItems(data.items);
        setTotal(data.total);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Could not load history.");
      });
  }, [page, debouncedSearch, prediction]);

  useEffect(load, [load]);

  const setPageSafe = (next: number, last: number) => {
    if (next < 1 || next > last) return;
    setPage(next);
  };

  if (error) return <ErrorState message={error} />;

  const lastPage = items ? Math.max(1, Math.ceil(total / PAGE_SIZE)) : 1;

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">
          Every analysis is stored locally with its probability distribution, model version
          and attention map. Filter and review past predictions.
        </p>
      </header>

      <div className="card section">
        <div className="flex gap-3" style={{ flexWrap: "wrap" }}>
          <form
            className="flex"
            style={{ minWidth: 220, flex: 1 }}
            onSubmit={(event) => event.preventDefault()}
          >
            <input
              type="search"
              className="input"
              placeholder="Search by filename or ID…"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(1);
              }}
              aria-label="Search analyses"
              style={{ width: "100%" }}
            />
          </form>
          <select
            className="input"
            value={prediction}
            onChange={(event) => {
              setPrediction(event.target.value);
              setPage(1);
            }}
            aria-label="Filter by prediction"
            style={{ maxWidth: 220 }}
          >
            <option value="">All predictions</option>
            <option value="glioma">Glioma</option>
            <option value="meningioma">Meningioma</option>
            <option value="no_tumor">No Tumor</option>
            <option value="pituitary_tumor">Pituitary Tumor</option>
          </select>
        </div>
      </div>

      {!items ? (
        <SkeletonRows rows={5} />
      ) : items.length === 0 ? (
        <EmptyState
          title={total === 0 ? "No analyses yet" : "No matches"}
          hint={
            total === 0
              ? "Upload a brain MRI image to create your first analysis."
              : "Nothing matches your filters. Try clearing the search or prediction filter."
          }
          action={
            total === 0 ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => navigate("/analysis/new")}
              >
                <Icon name="upload" size={16} /> Upload first image
              </button>
            ) : (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setSearch("");
                  setDebouncedSearch("");
                  setPrediction("");
                  setPage(1);
                }}
              >
                Clear filters
              </button>
            )
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
                  <th>Confidence</th>
                  <th>Model</th>
                  <th>When</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} onClick={() => navigate(`/analysis/${item.id}`)}>
                    <td style={{ whiteSpace: "nowrap" }}>{item.filename ?? item.id.slice(0, 8)}</td>
                    <td>
                      <span className={item.low_confidence ? "badge badge-warning" : "badge"}>
                        {labelOf(item.prediction)}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600 }}>{fmtPercent(item.confidence)}</td>
                    <td>
                      {item.model_name} <span className="text-faint text-xs">v{item.model_version}</span>
                    </td>
                    <td className="text-muted" title={fmtDateTime(item.created_at)}>
                      {fmtDateTime(item.created_at)}
                    </td>
                    <td>
                      <Link
                        to={`/analysis/${item.id}`}
                        className="btn btn-sm"
                        onClick={(event) => event.stopPropagation()}
                      >
                        View <Icon name="external" size={14} />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex-between p-3">
            <span className="text-sm text-muted">
              {total} result{total === 1 ? "" : "s"} · viewing {(page - 1) * PAGE_SIZE + 1}–
              {Math.min(page * PAGE_SIZE, total)}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                className="btn btn-sm"
                disabled={page <= 1}
                onClick={() => setPageSafe(page - 1, lastPage)}
              >
                Previous
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={page >= lastPage}
                onClick={() => setPageSafe(page + 1, lastPage)}
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

function labelOf(key: string): string {
  const map: Record<string, string> = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    no_tumor: "No Tumor",
    pituitary_tumor: "Pituitary Tumor",
  };
  return map[key] ?? key;
}