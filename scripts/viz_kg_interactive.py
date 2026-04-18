"""
Interactive KG Visualization Generator
Creates a self-contained HTML file with a D3.js force-directed graph.
Nodes are draggable, zoomable, with hover tooltips. Screenshot it for your poster.

Usage:
    python viz_kg_interactive.py

Reads:  data/processed/ml_research_kg.ttl
Saves:  output/kg_interactive.html  (open in browser)
"""

import json
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, RDFS

KG_PATH = "data/processed/ml_research_kg.ttl"
OUTPUT_PATH = "output/kg_interactive.html"

MLKG = Namespace("http://example.org/mlkg/")


def extract_subgraph_data(kg_path, max_pubs=15, max_authors_per_pub=3):
    """Extract a balanced subgraph and return as JSON-serializable dicts."""
    g = Graph()
    g.parse(kg_path, format="turtle")
    print(f"Loaded {len(g)} triples")

    # Build lookups
    entity_labels = {}
    entity_types = {}
    class_map = {
        str(MLKG.Publication): "Publication",
        str(MLKG.Author): "Author",
        str(MLKG.Institution): "Institution",
        str(MLKG.Venue): "Venue",
        str(MLKG.ResearchArea): "ResearchArea",
        str(MLKG.ResearchTopic): "ResearchTopic",
        str(MLKG.Dataset): "Dataset",
        str(MLKG.CodeRepository): "CodeRepository",
    }

    for s, _, o in g.triples((None, RDFS.label, None)):
        entity_labels[str(s)] = str(o)
    for s, _, o in g.triples((None, RDF.type, None)):
        if str(o) in class_map:
            entity_types[str(s)] = class_map[str(o)]

    # Build adjacency from object properties
    skip_preds = {
        str(MLKG.title), str(MLKG.abstract),
        str(MLKG.publicationYear), str(MLKG.citationCount),
        str(RDFS.label), str(RDFS.comment),
        str(RDF.type), str(RDFS.domain), str(RDFS.range),
    }

    adjacency = defaultdict(list)  # node -> [(neighbor, relation)]
    degree = defaultdict(int)

    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if p_str in skip_preds:
            continue
        if "www.w3.org" in s_str or "www.w3.org" in o_str:
            continue
        if not o_str.startswith("http"):
            continue
        if s_str not in entity_types or o_str not in entity_types:
            continue
        rel = p_str.split("/")[-1]
        adjacency[s_str].append((o_str, rel))
        adjacency[o_str].append((s_str, rel))
        degree[s_str] += 1
        degree[o_str] += 1

    # Select subgraph: top publications + capped authors + all their topics/areas
    pubs = [(n, degree[n]) for n in degree if entity_types.get(n) == "Publication"]
    pubs.sort(key=lambda x: -x[1])
    seed_pubs = [n for n, _ in pubs[:max_pubs]]

    selected = set(seed_pubs)
    edges = []
    seen_edges = set()

    for pub in seed_pubs:
        author_count = 0
        for nb, rel in adjacency[pub]:
            nb_type = entity_types.get(nb, "")

            if nb_type == "Author":
                if author_count >= max_authors_per_pub:
                    continue
                author_count += 1

            selected.add(nb)
            edge_key = tuple(sorted([pub, nb]))
            if edge_key not in seen_edges:
                edges.append({"source": pub, "target": nb, "relation": rel})
                seen_edges.add(edge_key)

    # Also add edges between selected non-pub nodes (e.g., author-area links)
    for node in list(selected):
        for nb, rel in adjacency[node]:
            if nb in selected:
                edge_key = tuple(sorted([node, nb]))
                if edge_key not in seen_edges:
                    edges.append({"source": node, "target": nb, "relation": rel})
                    seen_edges.add(edge_key)

    # Build node list
    nodes = []
    for n in selected:
        label = entity_labels.get(n, n.split("/")[-1])
        etype = entity_types.get(n, "Other")
        # Truncate long labels
        if etype == "Publication" and len(label) > 45:
            label = label[:42] + "..."
        nodes.append({
            "id": n,
            "label": label,
            "type": etype,
            "degree": degree.get(n, 0)
        })

    # Count types
    type_counts = defaultdict(int)
    for n in nodes:
        type_counts[n["type"]] += 1

    print(f"\nSubgraph: {len(nodes)} nodes, {len(edges)} edges")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Full graph stats
    full_stats = defaultdict(int)
    for t in entity_types.values():
        full_stats[t] += 1

    return nodes, edges, len(g), full_stats


def generate_html(nodes, edges, total_triples, full_stats, output_path):
    """Generate the self-contained HTML file."""

    data_json = json.dumps({"nodes": nodes, "links": edges}, indent=None)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Research Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: #FAFAF7;
    color: #1A1A2E;
    overflow: auto;
    width: 2800px;
    height: 4000px;
  }}

  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    background: rgba(250,250,247,0.92);
    backdrop-filter: blur(8px);
    padding: 14px 28px;
    border-bottom: 1px solid #E5E5E0;
    display: flex; justify-content: space-between; align-items: center;
  }}

  #header h1 {{
    font-size: 20px; font-weight: 700; color: #2E5E8A;
  }}

  #header .stats {{
    font-size: 13px; color: #6B7280;
  }}

  #legend {{
    position: fixed; bottom: 20px; left: 20px; z-index: 10;
    background: rgba(250,250,247,0.94);
    backdrop-filter: blur(8px);
    border: 1px solid #E5E5E0;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 12px;
  }}

  #legend h3 {{
    font-size: 13px; font-weight: 700; margin-bottom: 8px; color: #1A1A2E;
  }}

  .legend-item {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 5px;
  }}

  .legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
    border: 1.5px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    flex-shrink: 0;
  }}

  #tooltip {{
    position: fixed;
    background: rgba(26,26,46,0.92);
    color: #F3F4F6;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
    max-width: 320px;
    z-index: 20;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}

  #tooltip .tt-label {{
    font-weight: 700; font-size: 13px; margin-bottom: 3px;
  }}

  #tooltip .tt-type {{
    font-size: 11px; opacity: 0.7;
  }}

  svg {{ cursor: grab; }}
  svg:active {{ cursor: grabbing; }}
</style>
</head>
<body>

<div id="header">
  <h1>ML Research Knowledge Graph</h1>
  <div class="stats">
    {total_triples:,} triples &middot;
    {full_stats.get("Publication",0)} papers &middot;
    {full_stats.get("Author",0):,} authors &middot;
    {full_stats.get("ResearchTopic",0)} topics &middot;
    {full_stats.get("ResearchArea",0)} areas
  </div>
</div>

<div id="legend">
  <h3>Entity Type</h3>
</div>

<div id="tooltip">
  <div class="tt-label"></div>
  <div class="tt-type"></div>
</div>

<svg id="graph"></svg>

<script>
const data = {data_json};

const COLORS = {{
  Publication:    "#2E5E8A",
  Author:         "#D97706",
  Institution:    "#059669",
  Venue:          "#E85D04",
  ResearchArea:   "#7C3AED",
  ResearchTopic:  "#DB2777",
  Dataset:        "#0891B2",
  CodeRepository: "#65A30D",
  Other:          "#9CA3AF"
}};

const BASE_RADIUS = {{
  Publication:    10,
  Author:         5,
  ResearchArea:   16,
  ResearchTopic:  13,
  Venue:          14,
  Institution:    11,
  Dataset:        9,
  CodeRepository: 8,
}};

// Build legend
const typeCounts = {{}};
data.nodes.forEach(n => {{ typeCounts[n.type] = (typeCounts[n.type]||0)+1; }});
const legendEl = document.getElementById("legend");
Object.entries(typeCounts).sort((a,b) => b[1]-a[1]).forEach(([type, count]) => {{
  const item = document.createElement("div");
  item.className = "legend-item";
  item.innerHTML = `<div class="legend-dot" style="background:${{COLORS[type]||COLORS.Other}}"></div>${{type}} (${{count}})`;
  legendEl.appendChild(item);
}});

// SVG setup
const width = 2800;
const height = 4000;
const svg = d3.select("#graph").attr("width", width).attr("height", height);
const g = svg.append("g");

// Zoom
const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

// Tooltip
const tooltip = document.getElementById("tooltip");

// Force simulation
const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(65).strength(0.4))
  .force("charge", d3.forceManyBody().strength(-180).distanceMax(350))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(d => (BASE_RADIUS[d.type]||7) + 8))
  .force("x", d3.forceX(width/2).strength(0.04))
  .force("y", d3.forceY(height/2).strength(0.04));

// Draw edges
const link = g.append("g")
  .selectAll("line")
  .data(data.links)
  .join("line")
  .attr("stroke", d => {{
    const sType = (typeof d.source === 'object') ? d.source.type : data.nodes.find(n=>n.id===d.source)?.type;
    return COLORS[sType] || "#B0B8C8";
  }})
  .attr("stroke-opacity", 0.2)
  .attr("stroke-width", 1);

// Draw node halos
const halo = g.append("g")
  .selectAll("circle")
  .data(data.nodes)
  .join("circle")
  .attr("r", d => ((BASE_RADIUS[d.type]||7) + Math.min(d.degree, 15) * 0.5) * 1.8)
  .attr("fill", d => COLORS[d.type] || COLORS.Other)
  .attr("opacity", 0.08);

// Draw nodes
const node = g.append("g")
  .selectAll("circle")
  .data(data.nodes)
  .join("circle")
  .attr("r", d => (BASE_RADIUS[d.type]||7) + Math.min(d.degree, 15) * 0.5)
  .attr("fill", d => COLORS[d.type] || COLORS.Other)
  .attr("stroke", "#fff")
  .attr("stroke-width", 1.5)
  .attr("cursor", "pointer")
  .on("mouseover", (e, d) => {{
    tooltip.style.opacity = 1;
    tooltip.querySelector(".tt-label").textContent = d.label;
    tooltip.querySelector(".tt-type").textContent = d.type + " · degree " + d.degree;
    // Highlight connected
    link.attr("stroke-opacity", l =>
      (l.source.id===d.id || l.target.id===d.id) ? 0.6 : 0.05
    ).attr("stroke-width", l =>
      (l.source.id===d.id || l.target.id===d.id) ? 2.5 : 0.5
    );
    node.attr("opacity", n => {{
      if (n.id === d.id) return 1;
      const connected = data.links.some(l =>
        (l.source.id===d.id && l.target.id===n.id) ||
        (l.target.id===d.id && l.source.id===n.id)
      );
      return connected ? 1 : 0.15;
    }});
    halo.attr("opacity", n => n.id===d.id ? 0.15 : 0.03);
    label.attr("opacity", n => {{
      if (n.id === d.id) return 1;
      const connected = data.links.some(l =>
        (l.source.id===d.id && l.target.id===n.id) ||
        (l.target.id===d.id && l.source.id===n.id)
      );
      return connected ? 0.9 : 0.05;
    }});
  }})
  .on("mousemove", (e) => {{
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top = (e.clientY - 10) + "px";
  }})
  .on("mouseout", () => {{
    tooltip.style.opacity = 0;
    link.attr("stroke-opacity", 0.2).attr("stroke-width", 1);
    node.attr("opacity", 1);
    halo.attr("opacity", 0.08);
    label.attr("opacity", d => {{
      const r = BASE_RADIUS[d.type]||7;
      return (r >= 10 || d.degree >= 6) ? 0.9 : 0;
    }});
  }})
  .call(d3.drag()
    .on("start", (e,d) => {{ if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag", (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on("end", (e,d) => {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }})
  );

// Labels — only show for important nodes by default
const label = g.append("g")
  .selectAll("text")
  .data(data.nodes)
  .join("text")
  .text(d => {{
    if (d.label.length > 30) return d.label.slice(0,27) + "...";
    return d.label;
  }})
  .attr("font-size", d => {{
    if (d.type === "ResearchArea" || d.type === "ResearchTopic") return "11px";
    if (d.type === "Publication") return "9px";
    return "8.5px";
  }})
  .attr("font-weight", d => (d.type === "ResearchArea" || d.type === "ResearchTopic") ? "700" : "500")
  .attr("fill", d => COLORS[d.type] || "#1A1A2E")
  .attr("text-anchor", "middle")
  .attr("dy", d => -((BASE_RADIUS[d.type]||7) + Math.min(d.degree,15)*0.5 + 6))
  .attr("paint-order", "stroke")
  .attr("stroke", "#FAFAF7")
  .attr("stroke-width", 3.5)
  .attr("pointer-events", "none")
  .attr("opacity", d => {{
    const r = BASE_RADIUS[d.type]||7;
    return (r >= 10 || d.degree >= 6) ? 0.9 : 0;
  }});

// Simulation tick
simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  halo.attr("cx", d => d.x).attr("cy", d => d.y);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  label.attr("x", d => d.x).attr("y", d => d.y);
}});

// Initial zoom to fit after settling
setTimeout(() => {{
  const bounds = g.node().getBBox();
  const pad = 60;
  const scale = Math.min(
    width / (bounds.width + pad*2),
    height / (bounds.height + pad*2),
    1.5
  );
  const tx = width/2 - (bounds.x + bounds.width/2) * scale;
  const ty = height/2 - (bounds.y + bounds.height/2) * scale;
  svg.transition().duration(800).call(
    zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale)
  );
}}, 3000);
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nSaved interactive visualization to {output_path}")
    print("Open it in your browser!")


if __name__ == "__main__":
    nodes, edges, total_triples, full_stats = extract_subgraph_data(
        KG_PATH, max_pubs=15, max_authors_per_pub=3
    )
    generate_html(nodes, edges, total_triples, full_stats, OUTPUT_PATH)