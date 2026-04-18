"""
Interactive Embedding Visualizations
Creates self-contained HTML files with Plotly scatter plots.
Hover for entity names, zoom into clusters, toggle entity types.

Usage:
    python viz_embeddings_interactive.py

Reads:  output/entity_embeddings.json + data/processed/ml_research_kg.ttl
Saves:  output/embeddings_tsne_interactive.html
        output/embeddings_pca_interactive.html
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from rdflib import Graph, Namespace, RDF, RDFS

KG_PATH = "data/processed/ml_research_kg.ttl"
EMBEDDINGS_PATH = "output/entity_embeddings.json"
OUTPUT_DIR = "output"

MLKG = Namespace("http://example.org/mlkg/")

COLORS = {
    "Publication":    "#2E5E8A",
    "Author":         "#D97706",
    "Institution":    "#059669",
    "Venue":          "#E85D04",
    "ResearchArea":   "#7C3AED",
    "ResearchTopic":  "#DB2777",
    "Dataset":        "#0891B2",
    "CodeRepository": "#65A30D",
    "Other":          "#9CA3AF",
}

MARKER_SYMBOLS = {
    "Publication":    "circle",
    "Author":         "circle",
    "ResearchArea":   "diamond",
    "ResearchTopic":  "square",
    "Venue":          "triangle-up",
    "Institution":    "cross",
    "Dataset":        "hexagon",
    "CodeRepository": "star",
    "Other":          "circle",
}

MARKER_SIZES = {
    "Publication":    18,
    "Author":         18,
    "ResearchArea":   28,
    "ResearchTopic":  22,
    "Venue":          24,
    "Institution":    16,
    "Dataset":        16,
    "CodeRepository": 14,
    "Other":          8,
}

MAX_PER_TYPE = {
    "Publication":    200,
    "Author":         300,
    "ResearchArea":   None,
    "ResearchTopic":  None,
    "Venue":          None,
    "Institution":    None,
    "Dataset":        None,
    "CodeRepository": None,
}


def load_data():
    g = Graph()
    g.parse(KG_PATH, format="turtle")

    entity_types = {}
    entity_labels = {}

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
    for s, _, o in g.triples((None, RDF.type, None)):
        if str(o) in class_map:
            entity_types[str(s)] = class_map[str(o)]
    for s, _, o in g.triples((None, RDFS.label, None)):
        entity_labels[str(s)] = str(o)

    with open(EMBEDDINGS_PATH, "r") as f:
        raw_embeddings = json.load(f)

    print(f"Loaded {len(raw_embeddings)} embeddings, {len(entity_types)} typed entities")
    return raw_embeddings, entity_types, entity_labels


def sample_and_reduce(raw_embeddings, entity_types, entity_labels):
    """Sample proportionally, run t-SNE and PCA, return structured data."""
    by_type = defaultdict(list)
    for entity, emb in raw_embeddings.items():
        etype = entity_types.get(entity, "Other")
        by_type[etype].append((entity, emb))

    print("\nEntity types with embeddings:")
    for etype, items in sorted(by_type.items()):
        print(f"  {etype}: {len(items)}")

    rng = np.random.RandomState(42)
    sampled = []

    for etype, items in by_type.items():
        max_n = MAX_PER_TYPE.get(etype)
        if max_n is not None and len(items) > max_n:
            indices = rng.choice(len(items), max_n, replace=False)
            items = [items[i] for i in indices]
        for entity, emb in items:
            label = entity_labels.get(entity, entity.split("/")[-1])
            if len(label) > 50:
                label = label[:47] + "..."
            sampled.append({
                "entity": entity,
                "label": label,
                "type": etype,
                "emb": emb,
            })

    X = np.array([s["emb"] for s in sampled])
    print(f"\nSampled {len(X)} entities")

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(40, len(X)-1),
                random_state=42, max_iter=1500, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X)

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    print(f"  Variance explained: {var1:.1f}% + {var2:.1f}%")

    # Structure for Plotly
    results = {"tsne": [], "pca": []}
    for i, s in enumerate(sampled):
        entry = {"label": s["label"], "type": s["type"]}
        results["tsne"].append({**entry, "x": float(X_tsne[i, 0]), "y": float(X_tsne[i, 1])})
        results["pca"].append({**entry, "x": float(X_pca[i, 0]), "y": float(X_pca[i, 1])})

    return results, var1, var2


def generate_html(points, method, title_suffix, output_path):
    """Generate a self-contained Plotly HTML file."""

    # Group by type for traces
    by_type = defaultdict(list)
    for p in points:
        by_type[p["type"]].append(p)

    # Build Plotly traces
    traces = []
    # Draw order: big groups first so small ones are on top
    order = ["Author", "Publication", "Other", "CodeRepository", "Dataset",
             "Institution", "Venue", "ResearchTopic", "ResearchArea"]

    for etype in order:
        if etype not in by_type:
            continue
        pts = by_type[etype]
        traces.append({
            "x": [p["x"] for p in pts],
            "y": [p["y"] for p in pts],
            "text": [p["label"] for p in pts],
            "mode": "markers",
            "type": "scatter",
            "name": f"{etype} ({len(pts)})",
            "marker": {
                "color": COLORS.get(etype, "#9CA3AF"),
                "size": MARKER_SIZES.get(etype, 5),
                "symbol": MARKER_SYMBOLS.get(etype, "circle"),
                "opacity": 0.7 if etype in ("Author", "Publication") else 0.95,
                "line": {"width": 0.5, "color": "white"}
            },
            "hovertemplate": "<b>%{{text}}</b><br>" + etype + "<extra></extra>",
        })

    traces_json = json.dumps(traces, indent=None)
    method_upper = method.upper()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KG Embeddings — {method_upper}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'DM Sans', sans-serif;
    background: #FAFAF7;
    height: 1100px;
    width: 3000px;
    overflow: auto;
    display: flex;
    flex-direction: column;
  }}
  #plot {{ width: 3000px; height: 1100px; }}
</style>
</head>
<body>
<div id="plot"></div>
<script>
const traces = {traces_json};

const layout = {{
  title: {{
    text: "Knowledge Graph Entity Embeddings ({method_upper})",
    font: {{ family: "DM Sans", size: 46, color: "#2E5E8A" }},
    x: 0.5
  }},
  xaxis: {{
    title: {{ text: "{method_upper} Dimension 1", font: {{ size: 22, color: "#6B7280" }} }},
    gridcolor: "#E5E5E0",
    zerolinecolor: "#D1D5DB",
    tickfont: {{ color: "#6B7280", size: 40 }}
    }},
  yaxis: {{
    title: {{ text: "{method_upper} Dimension 2", font: {{ size: 22, color: "#6B7280" }} }},
    gridcolor: "#E5E5E0",
    zerolinecolor: "#D1D5DB",
    tickfont: {{ color: "#6B7280", size: 40 }}
}},
  paper_bgcolor: "#FAFAF7",
  plot_bgcolor: "#FAFAF7",
  legend: {{
    font: {{ size: 16 }},
    bgcolor: "rgba(250,250,247,0.9)",
    bordercolor: "#E5E5E0",
    borderwidth: 1
  }},
  hovermode: "closest",
  margin: {{ t: 60, b: 60, l: 60, r: 20 }},
  width: 3000,
  height: 1100
}};

const config = {{
  responsive: false,
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d'],
  displaylogo: false
}};

Plotly.newPlot("plot", traces, layout, config);
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved {method_upper} interactive plot to {output_path}")


if __name__ == "__main__":
    raw_embeddings, entity_types, entity_labels = load_data()
    results, var1, var2 = sample_and_reduce(raw_embeddings, entity_types, entity_labels)

    output_dir = Path(OUTPUT_DIR)

    generate_html(results["tsne"], "tsne",
                  "t-SNE", output_dir / "embeddings_tsne_interactive.html")

    generate_html(results["pca"], "pca",
                  f"PCA ({var1:.0f}%+{var2:.0f}% var)",
                  output_dir / "embeddings_pca_interactive.html")

    print("\nDone! Open the HTML files in your browser.")