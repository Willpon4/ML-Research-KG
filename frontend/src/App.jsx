import { useState, useEffect, useRef, useCallback } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Colors ───────────────────────────────────────────────────
const TYPE_COLORS = {
  Publication: "#3B82F6",
  Author: "#F59E0B",
  Institution: "#10B981",
  Venue: "#E85D04",
  ResearchArea: "#8B5CF6",
  ResearchTopic: "#EC4899",
  Dataset: "#06B6D4",
  CodeRepository: "#84CC16",
  Other: "#6B7280",
};

// ── Shared Components ────────────────────────────────────────

function TypeBadge({ type }) {
  return (
    <span
      style={{
        background: TYPE_COLORS[type] || TYPE_COLORS.Other,
        color: "#fff",
        padding: "2px 10px",
        borderRadius: "999px",
        fontSize: "11px",
        fontWeight: 600,
        letterSpacing: "0.03em",
      }}
    >
      {type}
    </span>
  );
}

function Loader() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "60px" }}>
      <div
        style={{
          width: 36,
          height: 36,
          border: "3px solid #1E293B",
          borderTop: "3px solid #3B82F6",
          borderRadius: "50%",
          animation: "spin 0.8s linear infinite",
        }}
      />
    </div>
  );
}

// ── Stats Page ───────────────────────────────────────────────

function StatsPage() {
  const [stats, setStats] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/stats`).then((r) => r.json()).then(setStats);
  }, []);

  if (!stats) return <Loader />;

  const entityOrder = ["Publication", "Author", "ResearchArea", "ResearchTopic", "Venue", "Institution", "Dataset", "CodeRepository"];
  const entities = entityOrder.filter((e) => stats.entity_counts[e]);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Knowledge Graph Overview</h2>
      <p style={{ color: "#94A3B8", marginBottom: 32 }}>
        A snapshot of the ML research knowledge graph.
      </p>

      {/* Big numbers */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 40 }}>
        {[
          { label: "Total Triples", value: stats.total_triples.toLocaleString() },
          { label: "Total Entities", value: stats.total_entities.toLocaleString() },
          { label: "Entity Types", value: entities.length },
        ].map((s) => (
          <div
            key={s.label}
            style={{
              background: "#1E293B",
              borderRadius: 12,
              padding: "28px 24px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 36, fontWeight: 700, color: "#3B82F6" }}>{s.value}</div>
            <div style={{ fontSize: 13, color: "#94A3B8", marginTop: 6 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Entity breakdown */}
      <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>Entity Breakdown</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {entities.map((type) => (
          <div
            key={type}
            style={{
              background: "#1E293B",
              borderRadius: 10,
              padding: "16px 18px",
              borderLeft: `4px solid ${TYPE_COLORS[type]}`,
            }}
          >
            <div style={{ fontSize: 22, fontWeight: 700 }}>
              {stats.entity_counts[type]?.toLocaleString()}
            </div>
            <div style={{ fontSize: 12, color: "#94A3B8", marginTop: 4 }}>{type}</div>
          </div>
        ))}
      </div>

      {/* Relationships */}
      <h3 style={{ fontSize: 18, fontWeight: 600, marginTop: 36, marginBottom: 16 }}>
        Relationships
      </h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
        {Object.entries(stats.relationship_counts)
          .sort((a, b) => b[1] - a[1])
          .map(([rel, count]) => (
            <div
              key={rel}
              style={{
                background: "#1E293B",
                borderRadius: 8,
                padding: "10px 16px",
                fontSize: 13,
              }}
            >
              <span style={{ color: "#94A3B8" }}>{rel}</span>{" "}
              <span style={{ fontWeight: 700, color: "#E2E8F0" }}>
                {count.toLocaleString()}
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}

// ── Search Page ──────────────────────────────────────────────

function SearchPage({ onSelectNode }) {
  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const doSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    const params = new URLSearchParams({ q: query, limit: "30" });
    if (typeFilter) params.set("type", typeFilter);
    const res = await fetch(`${API}/api/search?${params}`);
    const data = await res.json();
    setResults(data.results || []);
    setLoading(false);
  }, [query, typeFilter]);

  return (
    <div style={{ maxWidth: 800, margin: "0 auto" }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Search</h2>
      <p style={{ color: "#94A3B8", marginBottom: 24 }}>
        Find papers, authors, topics, and research areas in the knowledge graph.
      </p>

      <div style={{ display: "flex", gap: 10, marginBottom: 24 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && doSearch()}
          placeholder="Search papers, authors, topics..."
          style={{
            flex: 1,
            background: "#1E293B",
            border: "1px solid #334155",
            borderRadius: 10,
            padding: "12px 18px",
            color: "#E2E8F0",
            fontSize: 15,
            outline: "none",
          }}
        />
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          style={{
            background: "#1E293B",
            border: "1px solid #334155",
            borderRadius: 10,
            padding: "12px 14px",
            color: "#E2E8F0",
            fontSize: 13,
          }}
        >
          <option value="">All types</option>
          {Object.keys(TYPE_COLORS).filter((t) => t !== "Other").map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button
          onClick={doSearch}
          style={{
            background: "#3B82F6",
            border: "none",
            borderRadius: 10,
            padding: "12px 24px",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Search
        </button>
      </div>

      {loading && <Loader />}

      {results.map((node) => (
        <div
          key={node.id}
          onClick={() => onSelectNode(node.id)}
          style={{
            background: "#1E293B",
            borderRadius: 10,
            padding: "16px 20px",
            marginBottom: 10,
            cursor: "pointer",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#253348")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#1E293B")}
        >
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{node.label}</div>
            <TypeBadge type={node.type} />
          </div>
          <div style={{ color: "#94A3B8", fontSize: 13 }}>{node.degree} connections</div>
        </div>
      ))}
    </div>
  );
}

// ── Node Detail Panel ────────────────────────────────────────

function NodePanel({ nodeId, onClose, onNavigate }) {
  const [node, setNode] = useState(null);
  const [neighbors, setNeighbors] = useState(null);

  useEffect(() => {
    if (!nodeId) return;
    fetch(`${API}/api/node/${encodeURIComponent(nodeId)}`)
      .then((r) => r.json())
      .then(setNode);
    fetch(`${API}/api/neighbors/${encodeURIComponent(nodeId)}?limit=20`)
      .then((r) => r.json())
      .then(setNeighbors);
  }, [nodeId]);

  if (!nodeId) return null;
  if (!node) return <Loader />;

  const neighborNodes = neighbors?.nodes?.filter((n) => n.id !== nodeId) || [];

  return (
    <div
      style={{
        position: "fixed",
        right: 0,
        top: 0,
        bottom: 0,
        width: 420,
        background: "#0F172A",
        borderLeft: "1px solid #1E293B",
        padding: "24px",
        overflowY: "auto",
        zIndex: 100,
      }}
    >
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          background: "none",
          border: "none",
          color: "#94A3B8",
          fontSize: 22,
          cursor: "pointer",
        }}
      >
        ✕
      </button>

      <TypeBadge type={node.type} />
      <h3 style={{ fontSize: 20, fontWeight: 700, marginTop: 12, lineHeight: 1.3 }}>
        {node.label}
      </h3>

      {/* Properties */}
      {node.properties && Object.keys(node.properties).length > 0 && (
        <div style={{ marginTop: 20 }}>
          {Object.entries(node.properties).map(([key, val]) => (
            <div key={key} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {key}
              </div>
              <div style={{ fontSize: 13, color: "#CBD5E1", marginTop: 2, lineHeight: 1.5 }}>
                {val.length > 300 ? val.slice(0, 300) + "..." : val}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Neighbors */}
      <h4 style={{ fontSize: 14, fontWeight: 600, marginTop: 28, marginBottom: 12, color: "#94A3B8" }}>
        Connected Entities ({node.degree})
      </h4>
      {neighborNodes.map((nb) => (
        <div
          key={nb.id}
          onClick={() => onNavigate(nb.id)}
          style={{
            background: "#1E293B",
            borderRadius: 8,
            padding: "10px 14px",
            marginBottom: 6,
            cursor: "pointer",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: 13,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#253348")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#1E293B")}
        >
          <span style={{ fontWeight: 500 }}>
            {nb.label.length > 40 ? nb.label.slice(0, 37) + "..." : nb.label}
          </span>
          <TypeBadge type={nb.type} />
        </div>
      ))}
    </div>
  );
}

// ── Graph Explorer ───────────────────────────────────────────

function GraphPage({ selectedNode, onSelectNode }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [graphData, setGraphData] = useState(null);
  const [centerNode, setCenterNode] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [history, setHistory] = useState([]);

  // Search
  const doSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    const res = await fetch(`${API}/api/search?q=${encodeURIComponent(searchQuery)}&limit=8`);
    const data = await res.json();
    setSearchResults(data.results || []);
    setSearching(false);
  }, [searchQuery]);

  // Navigate to a node
  const navigateTo = useCallback((nodeId) => {
    if (centerNode) {
      setHistory((prev) => [...prev.slice(-10), centerNode.id]);
    }
    setSearchResults([]);
    setSearchQuery("");
    onSelectNode(nodeId);

    // Fetch node details + neighbors
    Promise.all([
      fetch(`${API}/api/node/${encodeURIComponent(nodeId)}`).then((r) => r.json()),
      fetch(`${API}/api/neighbors/${encodeURIComponent(nodeId)}?limit=50`).then((r) => r.json()),
    ]).then(([nodeDetail, neighbors]) => {
      setCenterNode(nodeDetail);
      setGraphData(neighbors);
    });
  }, [centerNode, onSelectNode]);

  const goBack = () => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    // Navigate without adding to history
    setSearchResults([]);
    onSelectNode(prev);
    Promise.all([
      fetch(`${API}/api/node/${encodeURIComponent(prev)}`).then((r) => r.json()),
      fetch(`${API}/api/neighbors/${encodeURIComponent(prev)}?limit=50`).then((r) => r.json()),
    ]).then(([nodeDetail, neighbors]) => {
      setCenterNode(nodeDetail);
      setGraphData(neighbors);
    });
  };

  // Load initial node if selectedNode is set
  useEffect(() => {
    if (selectedNode && !centerNode) {
      navigateTo(selectedNode);
    }
  }, [selectedNode]);

  // Render the graph
  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    import("https://cdn.jsdelivr.net/npm/d3@7/+esm").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = svgRef.current.clientWidth;
      const height = svgRef.current.clientHeight;
      const cx = width / 2;
      const cy = height / 2;

      // Background gradient
      const defs = svg.append("defs");
      const grad = defs.append("radialGradient").attr("id", "bg-glow");
      grad.append("stop").attr("offset", "0%").attr("stop-color", "#0F172A");
      grad.append("stop").attr("offset", "100%").attr("stop-color", "#050810");
      svg.append("rect").attr("width", width).attr("height", height).attr("fill", "url(#bg-glow)");

      const g = svg.append("g");

      // Zoom & pan
      const zoom = d3.zoom().scaleExtent([0.3, 6]).on("zoom", (e) => g.attr("transform", e.transform));
      svg.call(zoom);

      const centerId = graphData.center;
      const nodes = graphData.nodes.map((n) => ({
        ...n,
        isCenter: n.id === centerId,
      }));
      const links = graphData.edges.map((e) => ({
        source: e.source, target: e.target, relation: e.relation,
      }));

      function radius(d) {
        if (d.isCenter) return 28;
        const base = { Publication: 10, Author: 7, ResearchArea: 14, ResearchTopic: 12, Venue: 13, Institution: 9, Dataset: 9, CodeRepository: 8 };
        return (base[d.type] || 7) + Math.min(d.degree || 0, 20) * 0.2;
      }

      const sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d) => d.id).distance(100).strength(0.3))
        .force("charge", d3.forceManyBody().strength(-200).distanceMax(400))
        .force("center", d3.forceCenter(cx, cy))
        .force("collision", d3.forceCollide().radius((d) => radius(d) + 8))
        .force("radial", d3.forceRadial((d) => d.isCenter ? 0 : 180, cx, cy).strength((d) => d.isCenter ? 0.8 : 0.05));

      simRef.current = sim;

      // Edges
      const link = g.append("g").selectAll("line").data(links).join("line")
        .attr("stroke", (d) => {
          const sNode = nodes.find((n) => n.id === (typeof d.source === "object" ? d.source.id : d.source));
          return TYPE_COLORS[sNode?.type] || "#334155";
        })
        .attr("stroke-width", 1)
        .attr("stroke-opacity", 0.15);

      // Outer glow for center
      g.append("circle")
        .attr("cx", cx).attr("cy", cy).attr("r", 60)
        .attr("fill", TYPE_COLORS[centerNode?.type] || "#3B82F6")
        .attr("opacity", 0.06)
        .attr("class", "center-glow");

      g.append("circle")
        .attr("cx", cx).attr("cy", cy).attr("r", 40)
        .attr("fill", TYPE_COLORS[centerNode?.type] || "#3B82F6")
        .attr("opacity", 0.12)
        .attr("class", "center-glow");

      // Nodes
      const node = g.append("g").selectAll("circle").data(nodes).join("circle")
        .attr("r", radius)
        .attr("fill", (d) => TYPE_COLORS[d.type] || TYPE_COLORS.Other)
        .attr("stroke", (d) => d.isCenter ? "#fff" : "rgba(255,255,255,0.2)")
        .attr("stroke-width", (d) => d.isCenter ? 3 : 0.5)
        .attr("cursor", (d) => d.isCenter ? "default" : "pointer")
        .attr("opacity", (d) => d.isCenter ? 1 : 0.88)
        .on("click", (e, d) => { if (!d.isCenter) navigateTo(d.id); })
        .on("mouseover", function (e, d) {
          if (d.isCenter) return;
          d3.select(this).attr("r", radius(d) * 1.3).attr("opacity", 1);
          link.attr("stroke-opacity", (l) =>
            l.source.id === d.id || l.target.id === d.id ? 0.6 : 0.04
          );
        })
        .on("mouseout", function (e, d) {
          d3.select(this).attr("r", radius(d)).attr("opacity", 0.88);
          link.attr("stroke-opacity", 0.15);
        })
        .call(
          d3.drag()
            .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
        );

      // Labels
      const label = g.append("g").selectAll("text").data(nodes).join("text")
        .text((d) => {
          const maxLen = d.isCenter ? 40 : 22;
          return d.label.length > maxLen ? d.label.slice(0, maxLen - 3) + "..." : d.label;
        })
        .attr("font-size", (d) => d.isCenter ? "13px" : d.type === "ResearchArea" || d.type === "ResearchTopic" ? "10px" : "9px")
        .attr("font-weight", (d) => d.isCenter ? 700 : 500)
        .attr("fill", (d) => d.isCenter ? "#fff" : TYPE_COLORS[d.type] || "#94A3B8")
        .attr("text-anchor", "middle")
        .attr("dy", (d) => -(radius(d) + 8))
        .attr("pointer-events", "none")
        .attr("paint-order", "stroke")
        .attr("stroke", "#0B0F19")
        .attr("stroke-width", 3.5);

      sim.on("tick", () => {
        link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
        label.attr("x", (d) => d.x).attr("y", (d) => d.y);
        // Move glow with center node
        const cNode = nodes.find((n) => n.isCenter);
        if (cNode) {
          g.selectAll(".center-glow").attr("cx", cNode.x).attr("cy", cNode.y);
        }
      });

      // Auto-zoom to fit after settling
      setTimeout(() => {
        const bounds = g.node().getBBox();
        const pad = 80;
        const scale = Math.min(width / (bounds.width + pad * 2), height / (bounds.height + pad * 2), 1.5);
        const tx = width / 2 - (bounds.x + bounds.width / 2) * scale;
        const ty = height / 2 - (bounds.y + bounds.height / 2) * scale;
        svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
      }, 2000);
    });

    return () => { if (simRef.current) simRef.current.stop(); };
  }, [graphData]);

  // ── Landing state (no node selected) ──
  if (!centerNode) {
    return (
      <div style={{
        height: "calc(100vh - 60px)", background: "#0B0F19",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      }}>
        <div style={{
          width: 80, height: 80, borderRadius: 20,
          background: "linear-gradient(135deg, #3B82F6, #8B5CF6)",
          marginBottom: 28, opacity: 0.9,
        }} />
        <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8, color: "#E2E8F0" }}>
          Explore the Knowledge Graph
        </h2>
        <p style={{ color: "#64748B", marginBottom: 28, fontSize: 15 }}>
          Search for a paper, author, or topic to begin
        </p>

        <div style={{ display: "flex", gap: 10, width: 500, maxWidth: "90vw" }}>
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
            placeholder="e.g. transformer, attention, Yann LeCun..."
            style={{
              flex: 1, background: "#1E293B", border: "1px solid #334155",
              borderRadius: 12, padding: "14px 20px", color: "#E2E8F0",
              fontSize: 15, outline: "none",
            }}
            autoFocus
          />
          <button onClick={doSearch} style={{
            background: "#3B82F6", border: "none", borderRadius: 12,
            padding: "14px 24px", color: "#fff", fontWeight: 600, cursor: "pointer",
          }}>Search</button>
        </div>

        {/* Results */}
        {searchResults.length > 0 && (
          <div style={{
            marginTop: 12, width: 500, maxWidth: "90vw",
            background: "#1E293B", borderRadius: 12, overflow: "hidden",
            border: "1px solid #334155",
          }}>
            {searchResults.map((r) => (
              <div
                key={r.id}
                onClick={() => navigateTo(r.id)}
                style={{
                  padding: "12px 20px", cursor: "pointer",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  borderBottom: "1px solid #253348",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#253348")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div>
                  <div style={{ fontWeight: 500, fontSize: 14, marginBottom: 3 }}>
                    {r.label.length > 60 ? r.label.slice(0, 57) + "..." : r.label}
                  </div>
                  <TypeBadge type={r.type} />
                </div>
                <span style={{ color: "#64748B", fontSize: 12 }}>{r.degree} links</span>
              </div>
            ))}
          </div>
        )}

        {/* Quick start suggestions */}
        <div style={{ marginTop: 32, display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
          {["Transformers", "Deep Learning", "Attention Mechanisms", "Graph Neural Networks"].map((term) => (
            <button
              key={term}
              onClick={() => { setSearchQuery(term); setTimeout(() => {
                fetch(`${API}/api/search?q=${encodeURIComponent(term)}&limit=1`)
                  .then((r) => r.json())
                  .then((data) => { if (data.results?.[0]) navigateTo(data.results[0].id); });
              }, 0); }}
              style={{
                background: "transparent", border: "1px solid #334155",
                borderRadius: 999, padding: "8px 18px", color: "#94A3B8",
                fontSize: 13, cursor: "pointer",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "#1E293B"; e.currentTarget.style.color = "#E2E8F0"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#94A3B8"; }}
            >
              {term}
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ── Graph view ──
  return (
    <div style={{ position: "relative", flex: 1 }}>
      <svg ref={svgRef} style={{ width: "100%", height: "calc(100vh - 60px)" }} />

      {/* Top bar: back + search + info */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0,
        display: "flex", alignItems: "center", gap: 12,
        padding: "12px 16px",
        background: "linear-gradient(to bottom, rgba(5,8,16,0.85) 0%, transparent 100%)",
      }}>
        {history.length > 0 && (
          <button onClick={goBack} style={{
            background: "#1E293B", border: "1px solid #334155",
            borderRadius: 8, padding: "8px 14px", color: "#E2E8F0",
            fontSize: 13, cursor: "pointer", flexShrink: 0,
          }}>← Back</button>
        )}
        <button onClick={() => { setCenterNode(null); setGraphData(null); onSelectNode(null); setHistory([]); }} style={{
          background: "#1E293B", border: "1px solid #334155",
          borderRadius: 8, padding: "8px 14px", color: "#E2E8F0",
          fontSize: 13, cursor: "pointer", flexShrink: 0,
        }}>New Search</button>

        <div style={{ flex: 1 }} />

        <div style={{
          display: "flex", gap: 8, background: "#1E293B",
          borderRadius: 10, padding: "6px 8px", border: "1px solid #334155",
        }}>
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
            placeholder="Navigate to..."
            style={{
              background: "transparent", border: "none",
              color: "#E2E8F0", fontSize: 13, outline: "none", width: 180,
            }}
          />
          <button onClick={doSearch} style={{
            background: "#3B82F6", border: "none", borderRadius: 6,
            padding: "4px 12px", color: "#fff", fontSize: 12,
            fontWeight: 600, cursor: "pointer",
          }}>Go</button>
        </div>
      </div>

      {/* Mini search results dropdown */}
      {searchResults.length > 0 && (
        <div style={{
          position: "absolute", top: 54, right: 16,
          background: "#1E293B", borderRadius: 10,
          border: "1px solid #334155", width: 280, zIndex: 20,
          maxHeight: 300, overflowY: "auto",
        }}>
          {searchResults.map((r) => (
            <div key={r.id} onClick={() => navigateTo(r.id)}
              style={{ padding: "10px 14px", cursor: "pointer", borderBottom: "1px solid #253348", fontSize: 13 }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#253348")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div style={{ fontWeight: 500, marginBottom: 2 }}>
                {r.label.length > 35 ? r.label.slice(0, 32) + "..." : r.label}
              </div>
              <TypeBadge type={r.type} />
            </div>
          ))}
        </div>
      )}

      {/* Center node info card */}
      {centerNode && (
        <div style={{
          position: "absolute", bottom: 20, left: 20, right: 20,
          background: "rgba(15,23,42,0.92)", backdropFilter: "blur(12px)",
          borderRadius: 14, padding: "20px 24px",
          border: "1px solid #1E293B", maxWidth: 600,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
            <TypeBadge type={centerNode.type} />
            <span style={{ fontSize: 12, color: "#64748B" }}>{centerNode.degree} connections</span>
          </div>
          <h3 style={{ fontSize: 18, fontWeight: 700, lineHeight: 1.3, marginBottom: 6 }}>
            {centerNode.label}
          </h3>
          {centerNode.properties?.abstract && (
            <p style={{ fontSize: 12, color: "#94A3B8", lineHeight: 1.5 }}>
              {centerNode.properties.abstract.length > 250
                ? centerNode.properties.abstract.slice(0, 250) + "..."
                : centerNode.properties.abstract}
            </p>
          )}
          {centerNode.properties?.publicationYear && (
            <span style={{ fontSize: 12, color: "#64748B" }}>
              Published {centerNode.properties.publicationYear}
            </span>
          )}
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: "absolute", top: 60, right: 16,
        background: "rgba(15,23,42,0.85)", borderRadius: 10,
        padding: "12px 16px", border: "1px solid #1E293B",
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6, color: "#64748B" }}>Click a node to explore</div>
        {Object.entries(TYPE_COLORS).filter(([k]) => k !== "Other").map(([type, color]) => (
          <div key={type} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
            <span style={{ fontSize: 10, color: "#94A3B8" }}>{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function getRadius(d) {
  const base = {
    Publication: 6, Author: 4, ResearchArea: 12, ResearchTopic: 10,
    Venue: 11, Institution: 8, Dataset: 7, CodeRepository: 6,
  };
  return (base[d.type] || 5) + Math.min(d.degree, 20) * 0.3;
}

// ── Query Dashboard ──────────────────────────────────────────

function QueryPage() {
  const [useCases, setUseCases] = useState(null);
  const [selected, setSelected] = useState(null);
  const [param, setParam] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/queries`).then((r) => r.json()).then((data) => {
      setUseCases(data);
      const first = Object.keys(data)[0];
      setSelected(first);
      setParam(data[first]?.default_param || "");
    });
  }, []);

  const runQuery = async () => {
    if (!selected) return;
    setLoading(true);
    const params = new URLSearchParams();
    if (param) params.set("param", param);
    const res = await fetch(`${API}/api/query/${selected}?${params}`);
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  useEffect(() => {
    if (selected && useCases) {
      setParam(useCases[selected]?.default_param || "");
      setResult(null);
    }
  }, [selected]);

  if (!useCases) return <Loader />;

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto" }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>SPARQL Query Dashboard</h2>
      <p style={{ color: "#94A3B8", marginBottom: 24 }}>
        Explore the six use cases with live SPARQL queries against the knowledge graph.
      </p>

      {/* Use case buttons */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
        {Object.entries(useCases).map(([key, uc]) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            style={{
              background: selected === key ? "#3B82F6" : "#1E293B",
              border: selected === key ? "1px solid #3B82F6" : "1px solid #334155",
              borderRadius: 8,
              padding: "10px 18px",
              color: "#E2E8F0",
              fontSize: 13,
              fontWeight: selected === key ? 700 : 500,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            {uc.title}
          </button>
        ))}
      </div>

      {/* Selected use case */}
      {selected && useCases[selected] && (
        <div style={{ background: "#1E293B", borderRadius: 12, padding: "24px", marginBottom: 24 }}>
          <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>
            {useCases[selected].question}
          </div>
          <div style={{ color: "#94A3B8", fontSize: 14, marginBottom: 16 }}>
            {useCases[selected].description}
          </div>

          {useCases[selected].param_label && (
            <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
              <input
                value={param}
                onChange={(e) => setParam(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runQuery()}
                placeholder={useCases[selected].param_label}
                style={{
                  flex: 1,
                  background: "#0F172A",
                  border: "1px solid #334155",
                  borderRadius: 8,
                  padding: "10px 16px",
                  color: "#E2E8F0",
                  fontSize: 14,
                  outline: "none",
                }}
              />
            </div>
          )}

          <button
            onClick={runQuery}
            disabled={loading}
            style={{
              background: "#3B82F6",
              border: "none",
              borderRadius: 8,
              padding: "10px 28px",
              color: "#fff",
              fontWeight: 600,
              fontSize: 14,
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? "Running..." : "Run Query"}
          </button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          {/* Show SPARQL query */}
          <details style={{ marginBottom: 20 }}>
            <summary
              style={{
                cursor: "pointer",
                color: "#94A3B8",
                fontSize: 13,
                fontWeight: 600,
                marginBottom: 8,
              }}
            >
              View SPARQL Query
            </summary>
            <pre
              style={{
                background: "#0F172A",
                border: "1px solid #1E293B",
                borderRadius: 10,
                padding: 18,
                fontSize: 12,
                color: "#94A3B8",
                overflow: "auto",
                lineHeight: 1.6,
              }}
            >
              {result.query}
            </pre>
          </details>

          {result.error ? (
            <div style={{ color: "#EF4444", padding: 16 }}>Error: {result.error}</div>
          ) : (
            <>
              <div style={{ color: "#94A3B8", fontSize: 13, marginBottom: 12 }}>
                {result.count} result{result.count !== 1 ? "s" : ""}
              </div>

              {result.count > 0 && (
                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: 13,
                    }}
                  >
                    <thead>
                      <tr>
                        {result.columns.map((col) => (
                          <th
                            key={col}
                            style={{
                              textAlign: "left",
                              padding: "10px 14px",
                              borderBottom: "1px solid #334155",
                              color: "#94A3B8",
                              fontSize: 11,
                              textTransform: "uppercase",
                              letterSpacing: "0.06em",
                            }}
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.rows.map((row, i) => (
                        <tr
                          key={i}
                          style={{ borderBottom: "1px solid #1E293B" }}
                          onMouseEnter={(e) =>
                            (e.currentTarget.style.background = "#1E293B")
                          }
                          onMouseLeave={(e) =>
                            (e.currentTarget.style.background = "transparent")
                          }
                        >
                          {result.columns.map((col) => (
                            <td
                              key={col}
                              style={{
                                padding: "10px 14px",
                                color: "#CBD5E1",
                                maxWidth: 400,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                              }}
                            >
                              {row[col] ?? "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────

const PAGES = [
  { key: "graph", label: "Graph Explorer" },
  { key: "query", label: "Queries" },
  { key: "search", label: "Search" },
  { key: "stats", label: "Overview" },
];

export default function App() {
  const [page, setPage] = useState("graph");
  const [selectedNode, setSelectedNode] = useState(null);
  const [panelNode, setPanelNode] = useState(null);

  const handleSelectNode = (nodeId) => {
    setSelectedNode(nodeId);
    setPanelNode(nodeId);
    if (page !== "graph") setPage("graph");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0F172A", color: "#E2E8F0" }}>
      {/* Top nav */}
      <nav
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 24px",
          height: 60,
          background: "#0B0F19",
          borderBottom: "1px solid #1E293B",
          position: "sticky",
          top: 0,
          zIndex: 50,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 28,
              height: 28,
              borderRadius: 8,
              background: "linear-gradient(135deg, #3B82F6, #8B5CF6)",
            }}
          />
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.02em" }}>
            ML Research KG
          </span>
        </div>

        <div style={{ display: "flex", gap: 4 }}>
          {PAGES.map((p) => (
            <button
              key={p.key}
              onClick={() => setPage(p.key)}
              style={{
                background: page === p.key ? "#1E293B" : "transparent",
                border: "none",
                borderRadius: 6,
                padding: "8px 16px",
                color: page === p.key ? "#E2E8F0" : "#64748B",
                fontWeight: page === p.key ? 600 : 400,
                fontSize: 13,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        <div style={{ width: 120 }} />
      </nav>

      {/* Main content */}
      <main style={{ padding: page === "graph" ? 0 : "32px 24px" }}>
        {page === "stats" && <StatsPage />}
        {page === "search" && <SearchPage onSelectNode={handleSelectNode} />}
        {page === "query" && <QueryPage />}
        {page === "graph" && (
          <GraphPage selectedNode={selectedNode} onSelectNode={handleSelectNode} />
        )}
      </main>

      {/* Node detail panel */}
      <NodePanel
        nodeId={panelNode}
        onClose={() => setPanelNode(null)}
        onNavigate={handleSelectNode}
      />

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Söhne', -apple-system, BlinkMacSystemFont, sans-serif; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0F172A; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      `}</style>
    </div>
  );
}
