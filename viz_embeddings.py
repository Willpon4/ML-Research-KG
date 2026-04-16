"""
Improved t-SNE / PCA Embedding Visualizations — Poster Edition
Light-themed, samples proportionally across ALL entity types.

Usage:
    python viz_embeddings.py

Reads:  output/entity_embeddings.json + data/processed/ml_research_kg.ttl
Saves:  output/embeddings_tsne_v2.png, output/embeddings_pca_v2.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from rdflib import Graph, Namespace, RDF, RDFS

# ── Config ────────────────────────────────────────────────────
KG_PATH = "data/processed/ml_research_kg.ttl"
EMBEDDINGS_PATH = "output/entity_embeddings.json"
OUTPUT_DIR = "output"

MLKG = Namespace("http://example.org/mlkg/")

# Light poster theme
BG_COLOR = "#FAFAF7"
TEXT_COLOR = "#1A1A2E"
SUBTITLE_COLOR = "#6B7280"

# Colors matching poster palette
COLORS = {
    "Publication":    "#2E5E8A",
    "Author":         "#D97706",
    "Institution":    "#059669",
    "Venue":          "#DC2626",
    "ResearchArea":   "#7C3AED",
    "ResearchTopic":  "#DB2777",
    "Dataset":        "#0891B2",
    "CodeRepository": "#65A30D",
    "Other":          "#9CA3AF",
}

# Marker sizes and styles per type
MARKER_CONFIG = {
    "Publication":    {"s": 35,  "marker": "o", "alpha": 0.50},
    "Author":         {"s": 18,  "marker": "o", "alpha": 0.40},
    "ResearchArea":   {"s": 140, "marker": "D", "alpha": 0.95},
    "ResearchTopic":  {"s": 100, "marker": "s", "alpha": 0.95},
    "Venue":          {"s": 120, "marker": "^", "alpha": 0.95},
    "Institution":    {"s": 60,  "marker": "P", "alpha": 0.8},
    "Dataset":        {"s": 60,  "marker": "h", "alpha": 0.8},
    "CodeRepository": {"s": 40,  "marker": "v", "alpha": 0.7},
    "Other":          {"s": 15,  "marker": ".", "alpha": 0.3},
}

# Proportional sampling limits (None = include all)
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

    print(f"Loaded {len(raw_embeddings)} embeddings")
    print(f"Loaded {len(entity_types)} typed entities")

    return raw_embeddings, entity_types, entity_labels


def sample_entities(raw_embeddings, entity_types):
    by_type = {}
    for entity, emb in raw_embeddings.items():
        etype = entity_types.get(entity, "Other")
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append((entity, emb))

    print("\nEntity types with embeddings:")
    for etype, items in sorted(by_type.items()):
        print(f"  {etype}: {len(items)}")

    sampled_vectors = []
    sampled_types = []

    rng = np.random.RandomState(42)

    for etype, items in by_type.items():
        max_n = MAX_PER_TYPE.get(etype)
        if max_n is not None and len(items) > max_n:
            indices = rng.choice(len(items), max_n, replace=False)
            items = [items[i] for i in indices]

        for entity, emb in items:
            sampled_vectors.append(emb)
            sampled_types.append(etype)

    X = np.array(sampled_vectors)
    print(f"\nSampled {len(X)} entities for visualization")

    return X, sampled_types


def draw_embedding_plot(X_2d, types, method, output_path):
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw order: background types first, special types on top
    draw_order = ["Author", "Publication", "Other", "CodeRepository", "Dataset",
                  "Institution", "Venue", "ResearchTopic", "ResearchArea"]

    type_counts = {}
    for etype in draw_order:
        mask = [t == etype for t in types]
        indices = [i for i, m in enumerate(mask) if m]
        if not indices:
            continue

        type_counts[etype] = len(indices)
        cfg = MARKER_CONFIG.get(etype, MARKER_CONFIG["Other"])
        color = COLORS.get(etype, COLORS["Other"])

        # Soft halo for special types
        if etype in ("ResearchArea", "ResearchTopic", "Venue"):
            ax.scatter(
                X_2d[indices, 0], X_2d[indices, 1],
                c=color, s=cfg["s"] * 4, alpha=0.12,
                marker=cfg["marker"], edgecolors='none'
            )

        ax.scatter(
            X_2d[indices, 0], X_2d[indices, 1],
            c=color, s=cfg["s"],
            alpha=cfg["alpha"],
            marker=cfg["marker"],
            edgecolors='white', linewidth=0.3,
            zorder=draw_order.index(etype) + 2
        )

    # ── Legend ──
    legend_elements = []
    for etype in ["Publication", "Author", "ResearchArea", "ResearchTopic",
                   "Venue", "Institution", "Dataset", "CodeRepository"]:
        if etype in type_counts:
            cfg = MARKER_CONFIG.get(etype, MARKER_CONFIG["Other"])
            legend_elements.append(
                Line2D([0], [0], marker=cfg["marker"], color='none',
                       markerfacecolor=COLORS[etype], markersize=9,
                       markeredgecolor='white', markeredgewidth=0.3,
                       label=f"{etype} ({type_counts[etype]})")
            )

    legend = ax.legend(
        handles=legend_elements,
        loc='upper right', fontsize=10,
        frameon=True, framealpha=0.9,
        facecolor=BG_COLOR, edgecolor=SUBTITLE_COLOR,
        labelcolor=TEXT_COLOR,
        title="Entity Type", title_fontsize=11
    )
    legend.get_title().set_color(TEXT_COLOR)

    # ── Axes styling ──
    method_upper = method.upper()
    ax.set_xlabel(f"{method_upper} Dimension 1", fontsize=11,
                  color=SUBTITLE_COLOR, labelpad=10)
    ax.set_ylabel(f"{method_upper} Dimension 2", fontsize=11,
                  color=SUBTITLE_COLOR, labelpad=10)

    ax.tick_params(colors=SUBTITLE_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#D1D5DB')

    ax.grid(True, alpha=0.15, color='#9CA3AF', linestyle='--')

    # ── Title ──
    ax.set_title(
        f"Knowledge Graph Entity Embeddings ({method_upper})",
        fontsize=18, fontweight='bold', color="#2E5E8A",
        pad=20, fontfamily='sans-serif'
    )

    total = sum(type_counts.values())
    subtitle = f"TransE embeddings  ·  {total} entities  ·  {len(type_counts)} entity types"
    ax.text(
        0.5, -0.06, subtitle,
        transform=ax.transAxes, ha='center',
        fontsize=10, color=SUBTITLE_COLOR
    )

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f"Saved {method_upper} plot to {output_path}")


if __name__ == "__main__":
    raw_embeddings, entity_types, entity_labels = load_data()
    X, types = sample_entities(raw_embeddings, entity_types)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── t-SNE ──
    print("\nRunning t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(40, len(X) - 1),
        random_state=42,
        max_iter=1500,
        learning_rate='auto',
        init='pca'
    )
    X_tsne = tsne.fit_transform(X)
    draw_embedding_plot(X_tsne, types, "tsne", output_dir / "embeddings_tsne_v2.png")

    # ── PCA ──
    print("\nRunning PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    print(f"  PCA variance explained: {var1:.1f}% + {var2:.1f}% = {var1+var2:.1f}%")

    draw_embedding_plot(X_pca, types, "pca", output_dir / "embeddings_pca_v2.png")

    print("\nDone!")