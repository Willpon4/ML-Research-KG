"""
Impressive KG Subgraph Visualization — Poster Edition
Light-themed (matches poster), 14x20 inches, visible edges with glow nodes.

Usage:
    python viz_kg_subgraph.py

Reads:  data/processed/ml_research_kg.ttl
Saves:  output/kg_showcase.png
"""

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS

# ── Config ────────────────────────────────────────────────────
KG_PATH = "data/processed/ml_research_kg.ttl"
OUTPUT_PATH = "output/kg_showcase.png"

MLKG = Namespace("http://example.org/mlkg/")

# Light poster theme
BG_COLOR = "#FAFAF7"
TEXT_COLOR = "#1A1A2E"
SUBTITLE_COLOR = "#6B7280"
EDGE_COLOR = "#8B95A8"

# Rich saturated colors that read well on light backgrounds
COLORS = {
    "Publication":    "#2E5E8A",
    "Author":         "#D97706",
    "Institution":    "#059669",
    "Venue":          "#DC2626",
    "ResearchArea":   "#7C3AED",
    "ResearchTopic":  "#DB2777",
    "Dataset":        "#0891B2",
    "CodeRepository": "#65A30D",
}

# Base sizes per type (scaled by degree later)
BASE_SIZES = {
    "Publication":    220,
    "Author":          80,
    "ResearchArea":   350,
    "ResearchTopic":  280,
    "Venue":          300,
    "Institution":    240,
    "Dataset":        180,
    "CodeRepository": 140,
}


def load_kg(kg_path):
    g = Graph()
    g.parse(kg_path, format="turtle")

    entity_labels = {}
    entity_types = {}

    for s, _, o in g.triples((None, RDFS.label, None)):
        entity_labels[str(s)] = str(o)

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

    return g, entity_labels, entity_types


def build_nx_graph(g, entity_types):
    skip_preds = {
        str(MLKG.title), str(MLKG.abstract),
        str(MLKG.publicationYear), str(MLKG.citationCount),
        str(RDFS.label), str(RDFS.comment),
        str(RDF.type), str(RDFS.domain), str(RDFS.range),
    }

    G = nx.Graph()
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
        G.add_edge(s_str, o_str, relation=p_str.split("/")[-1])

    return G


def select_subgraph(G, entity_types, max_nodes=100):
    pub_nodes = [(n, G.degree(n)) for n in G.nodes()
                 if entity_types.get(n) == "Publication"]
    pub_nodes.sort(key=lambda x: -x[1])

    seeds = set(n for n, _ in pub_nodes[:10])

    expanded = set()
    for n in seeds:
        expanded.update(G.neighbors(n))

    all_nodes = seeds | expanded

    for etype in ["ResearchArea", "ResearchTopic", "Venue"]:
        type_nodes = [n for n in G.nodes() if entity_types.get(n) == etype]
        connected = [n for n in type_nodes if n in all_nodes]
        remaining = [n for n in type_nodes if n not in all_nodes]
        for n in (connected + remaining)[:5]:
            all_nodes.add(n)
            for nb in list(G.neighbors(n))[:3]:
                if nb in expanded or entity_types.get(nb) in ("Publication", "Author"):
                    all_nodes.add(nb)

    if len(all_nodes) > max_nodes:
        non_authors = [n for n in all_nodes if entity_types.get(n) != "Author"]
        authors = [(n, G.degree(n)) for n in all_nodes if entity_types.get(n) == "Author"]
        authors.sort(key=lambda x: -x[1])

        keep = set(non_authors)
        for n, _ in authors:
            if len(keep) >= max_nodes:
                break
            keep.add(n)
        all_nodes = keep

    return G.subgraph(all_nodes).copy()


def draw_showcase(SG, entity_types, entity_labels, g, output_path):
    fig, ax = plt.subplots(figsize=(14, 20), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── Layout ──
    try:
        pos = nx.kamada_kawai_layout(SG)
    except Exception:
        pos = nx.spring_layout(SG, k=3.0, iterations=150, seed=42)

    # ── Edges — tinted by connected node type, clearly visible ──
    edge_colors = []
    for u, v in SG.edges():
        u_type = entity_types.get(u, "Author")
        v_type = entity_types.get(v, "Author")
        priority = ["ResearchArea", "ResearchTopic", "Publication", "Venue", "Author"]
        for t in priority:
            if u_type == t or v_type == t:
                edge_colors.append(COLORS.get(t, EDGE_COLOR))
                break
        else:
            edge_colors.append(EDGE_COLOR)

    nx.draw_networkx_edges(
        SG, pos, ax=ax,
        edge_color=edge_colors,
        width=0.7, alpha=0.28,
        style='solid'
    )

    # ── Nodes with soft halo ──
    for etype, color in COLORS.items():
        nodes = [n for n in SG.nodes() if entity_types.get(n) == etype]
        if not nodes:
            continue

        base = BASE_SIZES.get(etype, 100)
        sizes = [base + SG.degree(n) * 30 for n in nodes]

        # Outer halo
        nx.draw_networkx_nodes(
            SG, pos, nodelist=nodes, ax=ax,
            node_color=color,
            node_size=[s * 2.2 for s in sizes],
            alpha=0.10, edgecolors='none'
        )

        # Inner halo
        nx.draw_networkx_nodes(
            SG, pos, nodelist=nodes, ax=ax,
            node_color=color,
            node_size=[s * 1.4 for s in sizes],
            alpha=0.18, edgecolors='none'
        )

        # Solid node
        nx.draw_networkx_nodes(
            SG, pos, nodelist=nodes, ax=ax,
            node_color=color,
            node_size=sizes,
            alpha=0.92,
            edgecolors='white', linewidths=1.2,
        )

    # ── Labels ──
    degrees = dict(SG.degree())
    label_nodes = set()

    pubs = sorted(
        [n for n in SG.nodes() if entity_types.get(n) == "Publication"],
        key=lambda n: -degrees.get(n, 0)
    )[:5]
    label_nodes.update(pubs)

    for n in SG.nodes():
        if entity_types.get(n) in ("ResearchArea", "ResearchTopic", "Venue"):
            label_nodes.add(n)

    authors = sorted(
        [n for n in SG.nodes() if entity_types.get(n) == "Author"],
        key=lambda n: -degrees.get(n, 0)
    )[:8]
    label_nodes.update(authors)

    for n in label_nodes:
        lbl = entity_labels.get(n, "")
        if not lbl:
            continue
        etype = entity_types.get(n, "")
        if etype == "Publication" and len(lbl) > 30:
            lbl = lbl[:27] + "..."
        elif len(lbl) > 25:
            lbl = lbl[:22] + "..."

        x, y = pos[n]
        color = COLORS.get(etype, TEXT_COLOR)
        fontsize = 8
        if etype in ("ResearchArea", "ResearchTopic"):
            fontsize = 10
        elif etype == "Publication":
            fontsize = 8.5

        ax.text(
            x, y + 0.03, lbl,
            fontsize=fontsize, color=color,
            fontweight='bold', ha='center', va='bottom',
            path_effects=[
                pe.withStroke(linewidth=4, foreground=BG_COLOR),
            ]
        )

    # ── Legend ──
    type_counts = {}
    for n in SG.nodes():
        t = entity_types.get(n, "Other")
        type_counts[t] = type_counts.get(t, 0) + 1

    legend_elements = []
    for etype in ["Publication", "Author", "ResearchArea", "ResearchTopic", "Venue",
                   "Institution", "Dataset", "CodeRepository"]:
        if etype in type_counts:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=COLORS[etype], markersize=10,
                       markeredgecolor='white', markeredgewidth=0.8,
                       label=f"{etype} ({type_counts[etype]})")
            )

    legend = ax.legend(
        handles=legend_elements, loc='lower left',
        fontsize=11, frameon=True, framealpha=0.9,
        facecolor=BG_COLOR, edgecolor=SUBTITLE_COLOR,
        labelcolor=TEXT_COLOR, title="Entity Type",
        title_fontsize=12
    )
    legend.get_title().set_color(TEXT_COLOR)

    # ── Title ──
    ax.set_title(
        "ML Research Knowledge Graph",
        fontsize=24, fontweight='bold', color="#2E5E8A",
        pad=30, fontfamily='sans-serif'
    )

    # ── Stats subtitle ──
    total_triples = len(g)
    all_types = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        cls = str(o)
        for k, v in {str(MLKG.Publication): "papers", str(MLKG.Author): "authors",
                      str(MLKG.ResearchTopic): "topics", str(MLKG.ResearchArea): "areas"}.items():
            if cls == k:
                all_types[v] = all_types.get(v, 0) + 1

    stats = (
        f"{total_triples:,} triples  ·  "
        f"{all_types.get('papers', 0)} papers  ·  "
        f"{all_types.get('authors', 0):,} authors  ·  "
        f"{all_types.get('topics', 0)} topics  ·  "
        f"{all_types.get('areas', 0)} research areas"
    )
    ax.text(
        0.5, -0.01, stats,
        transform=ax.transAxes, ha='center',
        fontsize=12, color=SUBTITLE_COLOR,
        fontfamily='sans-serif'
    )

    ax.axis('off')
    plt.tight_layout(pad=1.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    print(f"Saved showcase KG to {output_path}")


if __name__ == "__main__":
    print("Loading knowledge graph...")
    g, entity_labels, entity_types = load_kg(KG_PATH)
    print(f"  {len(g)} triples loaded")

    print("Building NetworkX graph...")
    G = build_nx_graph(g, entity_types)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Selecting subgraph...")
    SG = select_subgraph(G, entity_types, max_nodes=90)
    print(f"  Subgraph: {SG.number_of_nodes()} nodes, {SG.number_of_edges()} edges")

    type_counts = {}
    for n in SG.nodes():
        t = entity_types.get(n, "Other")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    print("Drawing showcase visualization...")
    draw_showcase(SG, entity_types, entity_labels, g, OUTPUT_PATH)
    print("Done!")