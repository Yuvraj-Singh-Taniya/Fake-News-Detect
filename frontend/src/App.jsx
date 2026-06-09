import { useState, useEffect, useCallback } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

// ── helpers ────────────────────────────────────────────────────────────────
const fmt = (n) => (typeof n === "number" ? (n * 100).toFixed(1) + "%" : "—");
const fmtDate = (s) => {
  if (!s) return "";
  const d = new Date(s);
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) +
    " · " + d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
};

const VERDICT_COLOR = {
  "Almost certainly fake": "#f43f5e",
  "Very likely fake": "#fb7185",
  "Likely fake": "#fb923c",
  "Possibly fake": "#fbbf24",
  "Uncertain — manual review recommended": "#94a3b8",
  "Possibly real": "#34d399",
  "Likely real": "#10b981",
  "Very likely real": "#059669",
  "Almost certainly real": "#047857",
};

// ── tab icons ──────────────────────────────────────────────────────────────
const TabIcon = ({ tab }) => {
  const icons = {
    detect: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/><path d="M11 8v6M8 11h6"/></svg>,
    history: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>,
    analytics: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>,
    live: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/></svg>,
  };
  return icons[tab] || null;
};

// ── Confidence Gauge ───────────────────────────────────────────────────────
function ConfidenceGauge({ value, label, isFake }) {
  const pct = Math.round(value * 100);
  const color = isFake ? "#f43f5e" : "#10b981";
  const r = 54;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={r} fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle cx="70" cy="70" r={r} fill="none" stroke={color} strokeWidth="12"
          strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(.4,0,.2,1)" }}
        />
        <text x="70" y="66" textAnchor="middle" fill="white" fontSize="26" fontWeight="700" fontFamily="'Sora', sans-serif">{pct}%</text>
        <text x="70" y="84" textAnchor="middle" fill="#64748b" fontSize="11" fontFamily="'Sora', sans-serif">confidence</text>
      </svg>
      <span style={{ fontSize: 13, fontWeight: 600, letterSpacing: 1.5, textTransform: "uppercase", color, fontFamily: "'Sora', sans-serif" }}>{label}</span>
    </div>
  );
}

// ── Feature Bar ───────────────────────────────────────────────────────────
function FeatureBar({ word, score, direction }) {
  const abs = Math.abs(score);
  const color = direction === "fake" ? "#f43f5e" : "#10b981";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
      <span style={{ width: 110, fontSize: 12, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace", textAlign: "right", flexShrink: 0 }}>{word}</span>
      <div style={{ flex: 1, background: "#0f172a", borderRadius: 3, overflow: "hidden", height: 8 }}>
        <div style={{ height: "100%", width: `${Math.min(100, abs * 800)}%`, background: color, borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ width: 50, fontSize: 11, color, fontFamily: "'JetBrains Mono', monospace" }}>{score > 0 ? "+" : ""}{score.toFixed(3)}</span>
    </div>
  );
}

// ── Fact Check Panel ──────────────────────────────────────────────────────
function FactCheckPanel({ factChecks, sourceCheck }) {
  const hasFactChecks = factChecks && factChecks.length > 0;
  const srcCount = sourceCheck?.found_in_sources ?? 0;
  const srcVerdict = sourceCheck?.source_verification ?? "";
  const topSources = sourceCheck?.top_sources ?? [];

  const srcColor =
    srcCount >= 5 ? "#10b981" :
    srcCount >= 2 ? "#fbbf24" :
    srcCount === 1 ? "#fb923c" :
    "#f43f5e";

  const ratingColor = (rating) => {
    if (!rating) return "#94a3b8";
    const r = rating.toLowerCase();
    if (r.includes("false") || r.includes("fake") || r.includes("incorrect") || r.includes("mislead")) return "#f43f5e";
    if (r.includes("true") || r.includes("correct") || r.includes("accurate")) return "#10b981";
    if (r.includes("mixed") || r.includes("partial") || r.includes("mostly")) return "#fbbf24";
    return "#94a3b8";
  };

  return (
    <div style={{ marginTop: 24, borderTop: "1px solid #1e293b", paddingTop: 20, display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Source Verification (Option 2) */}
      <div>
        <div style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 12, fontFamily: "'Sora', sans-serif" }}>
          🔍 Source Verification
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
          <span style={{
            fontFamily: "'Sora', sans-serif", fontSize: 22, fontWeight: 700, color: srcColor
          }}>{srcCount}</span>
          <span style={{ fontSize: 13, color: "#64748b", fontFamily: "'Sora', sans-serif" }}>
            news source{srcCount !== 1 ? "s" : ""} found
          </span>
        </div>
        <div style={{
          padding: "8px 14px", borderRadius: 8, fontSize: 12,
          background: `${srcColor}12`, border: `1px solid ${srcColor}30`,
          color: srcColor, fontFamily: "'Sora', sans-serif", marginBottom: topSources.length > 0 ? 10 : 0
        }}>
          {srcVerdict}
        </div>
        {topSources.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {topSources.map((s, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "7px 12px", background: "#0f172a",
                borderRadius: 8, border: "1px solid #1e293b"
              }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#475569", flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <a href={s.url} target="_blank" rel="noreferrer" style={{
                    fontSize: 12, color: "#818cf8", fontFamily: "'Sora', sans-serif",
                    textDecoration: "none", display: "block",
                    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis"
                  }}>
                    {s.name} ↗
                  </a>
                  {s.published && (
                    <span style={{ fontSize: 10, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
                      {new Date(s.published).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Google Fact Check (Option 1) */}
      <div>
        <div style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 12, fontFamily: "'Sora', sans-serif" }}>
          ✅ Fact Check Database
        </div>
        {!hasFactChecks ? (
          <div style={{
            padding: "8px 14px", borderRadius: 8, fontSize: 12,
            background: "#0f172a", border: "1px solid #1e293b",
            color: "#334155", fontFamily: "'Sora', sans-serif"
          }}>
            No matching claims found in fact-check databases
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {factChecks.map((fc, i) => {
              const rc = ratingColor(fc.rating);
              return (
                <div key={i} style={{
                  background: "#0f172a", borderRadius: 10,
                  border: `1px solid ${rc}25`, padding: "12px 14px"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10, marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: "#94a3b8", fontFamily: "'Sora', sans-serif", lineHeight: 1.5, flex: 1 }}>
                      {fc.claim}
                    </span>
                    {fc.rating && (
                      <span style={{
                        flexShrink: 0, fontSize: 11, fontWeight: 700,
                        color: rc, background: `${rc}15`,
                        border: `1px solid ${rc}35`,
                        padding: "3px 10px", borderRadius: 5,
                        fontFamily: "'Sora', sans-serif", whiteSpace: "nowrap"
                      }}>
                        {fc.rating}
                      </span>
                    )}
                  </div>
                  {fc.source && (
                    <a href={fc.url || "#"} target="_blank" rel="noreferrer" style={{
                      fontSize: 11, color: "#475569", fontFamily: "'Sora', sans-serif", textDecoration: "none"
                    }}>
                      {fc.source} ↗
                    </a>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}


function ResultCard({ result }) {
  const isFake = result.label === "fake";
  const color = VERDICT_COLOR[result.verdict] || (isFake ? "#f43f5e" : "#10b981");
  return (
    <div style={{
      background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
      border: `1px solid ${color}40`, borderRadius: 16, padding: "28px 32px",
      boxShadow: `0 0 40px ${color}20`, animation: "fadeSlide 0.4s ease"
    }}>
      <div style={{ display: "flex", gap: 32, alignItems: "center", flexWrap: "wrap" }}>
        <ConfidenceGauge value={result.confidence} label={result.label} isFake={isFake} />
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            background: `${color}18`, border: `1px solid ${color}50`,
            borderRadius: 8, padding: "6px 14px", marginBottom: 16
          }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "block" }} />
            <span style={{ color, fontWeight: 700, fontSize: 13, letterSpacing: 0.5, fontFamily: "'Sora', sans-serif" }}>{result.verdict}</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {Object.entries(result.probabilities).map(([k, v]) => (
              <div key={k} style={{ background: "#0f172a", borderRadius: 10, padding: "12px 16px", border: `1px solid #1e293b` }}>
                <div style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1, fontFamily: "'Sora', sans-serif", marginBottom: 4 }}>{k}</div>
                <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'Sora', sans-serif", color: k === result.label ? color : "#475569" }}>{fmt(v)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      {result.top_features && result.top_features.length > 0 && (
        <div style={{ marginTop: 24, borderTop: "1px solid #1e293b", paddingTop: 20 }}>
          <div style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 14, fontFamily: "'Sora', sans-serif" }}>Top Influential Words</div>
          {result.top_features.slice(0, 8).map((f, i) => <FeatureBar key={i} {...f} />)}
        </div>
      )}
      {(result.fact_checks !== undefined || result.source_check !== undefined) && (
        <FactCheckPanel
          factChecks={result.fact_checks || []}
          sourceCheck={result.source_check || {}}
        />
      )}
    </div>
  );
}

// ── Detect Tab ────────────────────────────────────────────────────────────
function DetectTab() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [explain, setExplain] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!title.trim() && !text.trim()) { setError("Please enter a title or article text."); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text, source, explain }),
      });
      const data = await res.json();
      if (data.error) setError(data.error);
      else setResult(data);
    } catch {
      setError("Could not connect to the backend. Make sure it's running.");
    } finally { setLoading(false); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gap: 14 }}>
        <div>
          <label style={labelStyle}>Article Headline</label>
          <input value={title} onChange={e => setTitle(e.target.value)}
            placeholder="Paste the news headline here…" style={inputStyle}
            onKeyDown={e => e.key === "Enter" && e.ctrlKey && handleSubmit()} />
        </div>
        <div>
          <label style={labelStyle}>Article Body <span style={{ color: "#334155" }}>(optional)</span></label>
          <textarea value={text} onChange={e => setText(e.target.value)}
            placeholder="Paste the article body here for better accuracy…" rows={6}
            style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 14, alignItems: "end" }}>
          <div>
            <label style={labelStyle}>Source URL <span style={{ color: "#334155" }}>(optional)</span></label>
            <input value={source} onChange={e => setSource(e.target.value)} placeholder="https://…" style={inputStyle} />
          </div>
          <label style={{
            display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
            padding: "12px 18px", background: "#0f172a", borderRadius: 10,
            border: `1px solid ${explain ? "#6366f1" : "#1e293b"}`,
            color: explain ? "#818cf8" : "#475569",
            fontFamily: "'Sora', sans-serif", fontSize: 13, fontWeight: 500,
            transition: "all 0.2s", userSelect: "none", whiteSpace: "nowrap"
          }}>
            <input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} style={{ display: "none" }} />
            <span style={{
              width: 16, height: 16, borderRadius: 4, border: `2px solid ${explain ? "#6366f1" : "#334155"}`,
              background: explain ? "#6366f1" : "transparent", display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s", flexShrink: 0
            }}>
              {explain && <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 5l2.5 2.5L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>}
            </span>
            Show explanations
          </label>
        </div>
      </div>
      <button onClick={handleSubmit} disabled={loading} style={{
        background: loading ? "#1e293b" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
        color: loading ? "#475569" : "white", border: "none", borderRadius: 12, padding: "15px 0",
        fontFamily: "'Sora', sans-serif", fontWeight: 700, fontSize: 15,
        cursor: loading ? "not-allowed" : "pointer", letterSpacing: 0.5, transition: "all 0.2s",
        boxShadow: loading ? "none" : "0 4px 24px #6366f140"
      }}>
        {loading ? (
          <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
            <span style={{ display: "inline-block", width: 16, height: 16, border: "2px solid #475569", borderTop: "2px solid #6366f1", borderRadius: "50%", animation: "spin 0.7s linear infinite" }} />
            Analyzing…
          </span>
        ) : "Analyze Article →"}
      </button>
      {error && (
        <div style={{ background: "#1e0a0f", border: "1px solid #f4435e60", borderRadius: 10, padding: "12px 16px", color: "#f43f5e", fontSize: 13, fontFamily: "'Sora', sans-serif" }}>
          ⚠ {error}
        </div>
      )}
      {result && <ResultCard result={result} />}
    </div>
  );
}

// ── Live News Tab ─────────────────────────────────────────────────────────
function LiveNewsTab() {
  const [articles, setArticles] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadLatest = async () => {
    setLoading(true); setError("");
    try {
      const res  = await fetch(`${API}/news/latest`);
      const data = await res.json();
      if (data.error) setError(data.error);
      else setArticles(data);
    } catch { setError("Could not fetch news. Check backend connection."); }
    finally { setLoading(false); }
  };

  const search = async () => {
    if (!query.trim()) return;
    setLoading(true); setError("");
    try {
      const res  = await fetch(`${API}/news/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      if (data.error) setError(data.error);
      else setArticles(data);
    } catch { setError("Could not fetch news. Check backend connection."); }
    finally { setLoading(false); }
  };

  useEffect(() => { loadLatest(); }, []);

  const fakeCount = articles.filter(a => a.label === "fake").length;
  const realCount = articles.filter(a => a.label === "real").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Search bar */}
      <div style={{ display: "flex", gap: 8 }}>
        <input value={query} onChange={e => setQuery(e.target.value)}
          placeholder="Search news topic… (e.g. climate, election, AI)"
          style={{ ...inputStyle, flex: 1 }}
          onKeyDown={e => e.key === "Enter" && search()} />
        <button onClick={search} style={{
          padding: "12px 18px", borderRadius: 10, border: "none",
          background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
          color: "white", cursor: "pointer",
          fontFamily: "'Sora', sans-serif", fontWeight: 600, fontSize: 13,
          whiteSpace: "nowrap"
        }}>Search</button>
        <button onClick={loadLatest} style={{
          padding: "12px 14px", borderRadius: 10,
          border: "1px solid #1e293b", background: "#0f172a",
          color: "#64748b", cursor: "pointer",
          fontFamily: "'Sora', sans-serif", fontSize: 12
        }}>Latest</button>
      </div>

      {/* Summary bar */}
      {articles.length > 0 && !loading && (
        <div style={{
          display: "flex", gap: 12, padding: "12px 16px",
          background: "#0f172a", borderRadius: 10, border: "1px solid #1e293b",
          flexWrap: "wrap"
        }}>
          <span style={{ fontSize: 12, color: "#475569", fontFamily: "'Sora', sans-serif" }}>
            {articles.length} articles analyzed
          </span>
          <span style={{ fontSize: 12, color: "#f43f5e", fontFamily: "'Sora', sans-serif" }}>
            ● {fakeCount} flagged fake
          </span>
          <span style={{ fontSize: 12, color: "#10b981", fontFamily: "'Sora', sans-serif" }}>
            ● {realCount} likely real
          </span>
        </div>
      )}

      {error && (
        <div style={{ background: "#1e0a0f", border: "1px solid #f4435e60", borderRadius: 10, padding: "12px 16px", color: "#f43f5e", fontSize: 13, fontFamily: "'Sora', sans-serif" }}>
          ⚠ {error}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: "center", padding: 60, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
          <div style={{ display: "inline-block", width: 24, height: 24, border: "2px solid #1e293b", borderTop: "2px solid #6366f1", borderRadius: "50%", animation: "spin 0.7s linear infinite", marginBottom: 12 }} />
          <div>Fetching and analyzing live news…</div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {articles.map((a, i) => {
            const isFake = a.label === "fake";
            const isUncertain = a.verdict?.includes("Uncertain");
            const color = isUncertain ? "#94a3b8" : isFake ? "#f43f5e" : "#10b981";
            const published = a.published ? new Date(a.published).toLocaleDateString("en-IN", { day: "2-digit", month: "short" }) : "";
            let hostname = "";
            try { hostname = new URL(a.source).hostname.replace("www.", ""); } catch {}

            return (
              <div key={i} style={{
                background: "#0f172a", borderRadius: 12,
                border: `1px solid ${color}25`, padding: "14px 18px",
                display: "flex", gap: 14, alignItems: "flex-start",
                transition: "border-color 0.2s"
              }}>
                {a.image && (
                  <img src={a.image} alt="" style={{
                    width: 72, height: 54, borderRadius: 8,
                    objectFit: "cover", flexShrink: 0
                  }} onError={e => e.target.style.display = "none"} />
                )}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontFamily: "'Sora', sans-serif", fontWeight: 600,
                    fontSize: 13, color: "#e2e8f0", marginBottom: 8,
                    lineHeight: 1.5
                  }}>{a.title}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{
                      fontSize: 11, fontWeight: 700, color,
                      background: `${color}15`, border: `1px solid ${color}35`,
                      padding: "3px 10px", borderRadius: 5,
                      fontFamily: "'Sora', sans-serif", whiteSpace: "nowrap"
                    }}>
                      {isUncertain ? "UNCERTAIN" : a.label?.toUpperCase()} · {Math.round(a.confidence * 100)}%
                    </span>
                    <span style={{ fontSize: 11, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
                      {a.verdict}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
                    {hostname && (
                      <a href={a.source} target="_blank" rel="noreferrer" style={{
                        fontSize: 11, color: "#475569",
                        fontFamily: "'Sora', sans-serif", textDecoration: "none"
                      }}>
                        {hostname} ↗
                      </a>
                    )}
                    {published && (
                      <span style={{ fontSize: 11, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
                        {published}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!loading && articles.length === 0 && !error && (
        <div style={{ textAlign: "center", padding: 60, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>📡</div>
          No articles loaded yet.
        </div>
      )}
    </div>
  );
}

// ── History Tab ───────────────────────────────────────────────────────────
function HistoryTab() {
  const [records, setRecords] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const q = new URLSearchParams({ page, per_page: 10, ...(filter ? { label: filter } : {}) });
      const res = await fetch(`${API}/history?${q}`);
      const data = await res.json();
      setRecords(data.results || []);
      setTotal(data.total || 0);
      setPages(data.pages || 1);
    } catch { setRecords([]); }
    finally { setLoading(false); }
  }, [page, filter]);

  useEffect(() => { load(); }, [load]);

  const del = async (id) => {
    if (!window.confirm("Delete this record?")) return;
    await fetch(`${API}/history/${id}`, { method: "DELETE" });
    load();
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
        <span style={{ color: "#64748b", fontFamily: "'Sora', sans-serif", fontSize: 13 }}>{total} total predictions</span>
        <div style={{ display: "flex", gap: 8 }}>
          {["", "fake", "real"].map(v => (
            <button key={v} onClick={() => { setFilter(v); setPage(1); }} style={{
              padding: "6px 14px", borderRadius: 8,
              background: filter === v ? (v === "fake" ? "#f43f5e20" : v === "real" ? "#10b98120" : "#6366f120") : "#0f172a",
              color: filter === v ? (v === "fake" ? "#f43f5e" : v === "real" ? "#10b981" : "#818cf8") : "#475569",
              cursor: "pointer", fontFamily: "'Sora', sans-serif", fontSize: 12, fontWeight: 500,
              border: `1px solid ${filter === v ? (v === "fake" ? "#f43f5e40" : v === "real" ? "#10b98140" : "#6366f140") : "#1e293b"}`
            }}>{v || "All"}</button>
          ))}
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 40, color: "#334155" }}>Loading…</div>
      ) : records.length === 0 ? (
        <div style={{ textAlign: "center", padding: 60, color: "#334155", fontFamily: "'Sora', sans-serif" }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>📭</div>
          No predictions yet. Go detect some news!
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {records.map((r) => {
            const isFake = r.label === "fake";
            const color = isFake ? "#f43f5e" : "#10b981";
            const open = expanded === r._id;
            return (
              <div key={r._id} style={{ background: "#0f172a", borderRadius: 12, border: `1px solid #1e293b`, overflow: "hidden" }}>
                <div onClick={() => setExpanded(open ? null : r._id)}
                  style={{ padding: "14px 18px", cursor: "pointer", display: "flex", alignItems: "center", gap: 14 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontFamily: "'Sora', sans-serif", fontWeight: 600, fontSize: 14, color: "#e2e8f0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {r.title || r.text?.slice(0, 80) || "Untitled"}
                    </div>
                    <div style={{ fontSize: 11, color: "#334155", marginTop: 3, fontFamily: "'Sora', sans-serif" }}>
                      {fmtDate(r.created_at?.$date || r.created_at)} &nbsp;·&nbsp; {r.source || "No source"}
                    </div>
                  </div>
                  <div style={{ flexShrink: 0, textAlign: "right" }}>
                    <div style={{ color, fontWeight: 700, fontFamily: "'Sora', sans-serif", fontSize: 13 }}>
                      {r.label?.toUpperCase()} · {Math.round(r.confidence * 100)}%
                    </div>
                  </div>
                  <button onClick={e => { e.stopPropagation(); del(r._id); }} style={{
                    background: "none", border: "none", color: "#334155", cursor: "pointer", padding: 4, fontSize: 14
                  }} title="Delete">✕</button>
                </div>
                {open && (
                  <div style={{ padding: "14px 18px", borderTop: "1px solid #1e293b" }}>
                    {r.text && <p style={{ color: "#64748b", fontSize: 13, fontFamily: "'Sora', sans-serif", lineHeight: 1.6, margin: "0 0 10px" }}>{r.text.slice(0, 300)}{r.text.length > 300 ? "…" : ""}</p>}
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                      {Object.entries(r.probabilities || {}).map(([k, v]) => (
                        <span key={k} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 12, background: "#1e293b", color: k === r.label ? color : "#475569", fontFamily: "'Sora', sans-serif" }}>
                          {k}: {fmt(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {pages > 1 && (
        <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 20 }}>
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} style={pageBtnStyle(page === 1)}>← Prev</button>
          <span style={{ padding: "8px 14px", color: "#64748b", fontFamily: "'Sora', sans-serif", fontSize: 13 }}>{page} / {pages}</span>
          <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages} style={pageBtnStyle(page === pages)}>Next →</button>
        </div>
      )}
    </div>
  );
}

// ── Analytics Tab ──────────────────────────────────────────────────────────
function AnalyticsTab() {
  const [data, setData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/analytics`).then(r => r.json()).catch(() => null),
      fetch(`${API}/metrics`).then(r => r.json()).catch(() => null),
    ]).then(([a, m]) => { setData(a); setMetrics(m); setLoading(false); });
  }, []);

  if (loading) return <div style={{ textAlign: "center", padding: 60, color: "#334155" }}>Loading analytics…</div>;

  const trend = data?.recent_trend || [];
  const maxTrend = Math.max(...trend.map(t => t.total), 1);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
        {[
          { label: "Total Scans", value: data?.total || 0, color: "#818cf8" },
          { label: "Fake Detected", value: data?.fake_count || 0, color: "#f43f5e" },
          { label: "Real Detected", value: data?.real_count || 0, color: "#10b981" },
          { label: "Avg Confidence", value: data?.avg_confidence ? fmt(data.avg_confidence) : "—", color: "#fbbf24" },
        ].map(c => (
          <div key={c.label} style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "18px 20px" }}>
            <div style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Sora', sans-serif", marginBottom: 8 }}>{c.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: c.color, fontFamily: "'Sora', sans-serif" }}>{c.value}</div>
          </div>
        ))}
      </div>

      {data?.total > 0 && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "20px 24px" }}>
          <div style={{ fontSize: 12, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Sora', sans-serif", marginBottom: 14 }}>Detection Ratio</div>
          <div style={{ display: "flex", gap: 4, height: 20, borderRadius: 6, overflow: "hidden" }}>
            <div style={{ flex: data.fake_pct, background: "#f43f5e", transition: "flex 0.6s ease" }} />
            <div style={{ flex: data.real_pct, background: "#10b981", transition: "flex 0.6s ease" }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
            <span style={{ color: "#f43f5e", fontSize: 13, fontFamily: "'Sora', sans-serif" }}>Fake {data.fake_pct}%</span>
            <span style={{ color: "#10b981", fontSize: 13, fontFamily: "'Sora', sans-serif" }}>Real {data.real_pct}%</span>
          </div>
        </div>
      )}

      <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "20px 24px" }}>
        <div style={{ fontSize: 12, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Sora', sans-serif", marginBottom: 18 }}>7-Day Trend</div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 80 }}>
          {trend.map((t, i) => (
            <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4, height: "100%", justifyContent: "flex-end" }}>
              <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 1 }}>
                <div style={{ height: `${(t.fake / maxTrend) * 60}px`, background: "#f43f5e", borderRadius: "3px 3px 0 0", minHeight: t.fake > 0 ? 3 : 0 }} />
                <div style={{ height: `${(t.real / maxTrend) * 60}px`, background: "#10b981", borderRadius: t.fake === 0 ? "3px 3px 0 0" : 0, minHeight: t.real > 0 ? 3 : 0 }} />
              </div>
              <span style={{ fontSize: 10, color: "#334155", fontFamily: "'Sora', sans-serif", whiteSpace: "nowrap" }}>{t.date}</span>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 14 }}>
          {[["#f43f5e", "Fake"], ["#10b981", "Real"]].map(([c, l]) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: "block" }} />
              <span style={{ fontSize: 12, color: "#475569", fontFamily: "'Sora', sans-serif" }}>{l}</span>
            </div>
          ))}
        </div>
      </div>

      {metrics && (
        <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: "20px 24px" }}>
          <div style={{ fontSize: 12, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Sora', sans-serif", marginBottom: 16 }}>Model Performance</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {Object.entries(metrics).map(([name, m]) => (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 14px", background: "#060f1e", borderRadius: 8 }}>
                <span style={{ width: 160, fontSize: 12, color: "#64748b", fontFamily: "'Sora', sans-serif", textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</span>
                <div style={{ flex: 1, display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {["accuracy", "f1", "roc_auc"].map(k => m[k] !== undefined && (
                    <span key={k} style={{
                      fontSize: 12, fontFamily: "'Sora', sans-serif",
                      color: m[k] >= 0.99 ? "#10b981" : m[k] >= 0.9 ? "#fbbf24" : "#f43f5e",
                      background: "#0f172a", padding: "3px 10px", borderRadius: 5,
                    }}>
                      {k.replace(/_/, " ")}: {(m[k] * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Shared Styles ─────────────────────────────────────────────────────────
const inputStyle = {
  width: "100%", boxSizing: "border-box",
  background: "#0f172a", border: "1px solid #1e293b",
  borderRadius: 10, padding: "12px 16px",
  color: "#e2e8f0", fontFamily: "'Sora', sans-serif", fontSize: 14,
  outline: "none", transition: "border-color 0.2s",
};

const labelStyle = {
  display: "block", marginBottom: 8, fontSize: 12, color: "#475569",
  textTransform: "uppercase", letterSpacing: 1.2,
  fontFamily: "'Sora', sans-serif", fontWeight: 600,
};

const pageBtnStyle = (disabled) => ({
  padding: "8px 16px", borderRadius: 8, border: "1px solid #1e293b",
  background: "#0f172a", color: disabled ? "#1e293b" : "#64748b",
  cursor: disabled ? "not-allowed" : "pointer",
  fontFamily: "'Sora', sans-serif", fontSize: 13
});

// ── App Root ──────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("detect");
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => null);
  }, []);

  const tabs = [
    { id: "detect",    label: "Detect" },
    { id: "live",      label: "Live News" },
    { id: "history",   label: "History" },
    { id: "analytics", label: "Analytics" },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #060f1e; color: #e2e8f0; min-height: 100vh; }
        input:focus, textarea:focus { border-color: #6366f1 !important; outline: none; }
        input::placeholder, textarea::placeholder { color: #334155; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeSlide { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
      `}</style>

      <div style={{ maxWidth: 780, margin: "0 auto", padding: "24px 16px 80px" }}>
        {/* Header */}
        <div style={{ marginBottom: 32, paddingBottom: 24, borderBottom: "1px solid #0f172a" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                <div style={{
                  width: 36, height: 36, borderRadius: 10,
                  background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "0 4px 16px #6366f140"
                }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10 21h7a2 2 0 0 0 2-2V9.414a1 1 0 0 0-.293-.707l-5.414-5.414A1 1 0 0 0 12.586 3H7a2 2 0 0 0-2 2v11m0 5 1.5-1.5L8 18l1.5-1.5L11 18"/>
                  </svg>
                </div>
                <h1 style={{
                  fontFamily: "'Sora', sans-serif", fontWeight: 700, fontSize: 22,
                  background: "linear-gradient(135deg, #e2e8f0, #818cf8)",
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
                }}>TruthLens</h1>
              </div>
              <p style={{ color: "#334155", fontSize: 13, fontFamily: "'Sora', sans-serif" }}>
                AI-powered fake news detection · Powered by ML
              </p>
            </div>

            {health && (
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[
                  { label: "API",   ok: health.status === "ok" },
                  { label: "Model", ok: health.model_ready },
                  { label: "DB",    ok: health.database === "connected" },
                ].map(s => (
                  <span key={s.label} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "5px 10px", borderRadius: 6, fontSize: 11,
                    background: s.ok ? "#052010" : "#1a0510",
                    border: `1px solid ${s.ok ? "#10b98130" : "#f43f5e30"}`,
                    color: s.ok ? "#10b981" : "#f43f5e",
                    fontFamily: "'Sora', sans-serif", fontWeight: 500
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: s.ok ? "#10b981" : "#f43f5e",
                      animation: s.ok ? "pulse 2s infinite" : "none"
                    }} />
                    {s.label}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 28, background: "#0a1628", borderRadius: 12, padding: 4 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex: 1, padding: "10px 0",
              background: tab === t.id ? "#0f172a" : "transparent",
              border: tab === t.id ? "1px solid #1e293b" : "1px solid transparent",
              borderRadius: 9, cursor: "pointer",
              color: tab === t.id ? (t.id === "live" ? "#f87171" : "#e2e8f0") : "#334155",
              fontFamily: "'Sora', sans-serif", fontWeight: tab === t.id ? 600 : 400, fontSize: 12,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
              transition: "all 0.2s",
              boxShadow: tab === t.id ? "0 2px 8px #00000040" : "none"
            }}>
              <TabIcon tab={t.id} />
              {t.id === "live" && <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#f87171", animation: "pulse 1.5s infinite" }} />}
              {t.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {tab === "detect"    && <DetectTab />}
        {tab === "live"      && <LiveNewsTab />}
        {tab === "history"   && <HistoryTab />}
        {tab === "analytics" && <AnalyticsTab />}
      </div>
    </>
  );
}