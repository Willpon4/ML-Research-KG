"""
Knowledge Graph Network Visualizations
Renders the KG as a visual network using NetworkX.
Produces the "graphical abstract" for the poster.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

from rdflib import Graph as RDFGraph, Namespace, RDF, RDFS

MLKG = Namespace("http://example.org/mlkg/")

# Entity type colors (matching the poster palette)
TYPE_COLORS = {
    "Publication": "#2E5E8A",      # Deep blue
    "Author": "#D97706",            # Amber (poster accent color)
    "Institution": "#059669",       # Green
    "Venue": "#DC2626",             # Red
    "ResearchArea": "#7C3AED",      # Purple
    "ResearchTopic": "#DB2777",     # Pink
    "Dataset": "#0891B2",           # Cyan
    "CodeRepository": "#65A30D",    # Lime
}

TYPE_SIZES = {
    "Publication": 250,
    "Author": 100,
    "Institution": 200,
    "Venue": 200,
    "ResearchArea": 350,
    "ResearchTopic": 250,
    "Dataset": 150,
    "CodeRepository": 100,
}


def build_networkx_from_rdf(rdf_path):
    """Convert RDF graph to NetworkX graph with entity types."""
    rdf_g = RDFGraph()
    rdf_g.parse(rdf_path, format="turtle")

    # Build entity type map
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

    for s, _, o in rdf_g.triples((None, RDF.type, None)):
        if str(o) in class_map:
            entity_types[str(s)] = class_map[str(o)]

    # Build label map
    labels = {}
    for s, _, o in rdf_g.triples((None, RDFS.label, None)):
        labels[str(s)] = str(o)

    # Build NetworkX graph (only object property triples)
    G = nx.MultiDiGraph()

    skip_preds = {
        str(RDF.type), str(RDFS.label), str(RDFS.comment),
        str(RDFS.domain), str(RDFS.range), str(RDFS.subClassOf),
        str(MLKG.title), str(MLKG.abstract),
        str(MLKG.publicationYear), str(MLKG.citationCount),
    }

    for s, p, o in rdf_g:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if p_str in skip_preds:
            continue
        if not o_str.startswith("http"):
            continue
        if "www.w3.org" in s_str or "www.w3.org" in o_str:
            continue
        if s_str not in entity_types or o_str not in entity_types:
            continue

        # Add nodes with type
        G.add_node(s_str, type=entity_types[s_str],
                   label=labels.get(s_str, s_str.split("/")[-1]))
        G.add_node(o_str, type=entity_types[o_str],
                   label=labels.get(o_str, o_str.split("/")[-1]))

        # Edge with relation name
        rel_name = p_str.split("/")[-1]
        G.add_edge(s_str, o_str, relation=rel_name)

    return G, entity_types, labels


def visualize_full_graph(rdf_path, output_path="output/kg_network.png",
                          figsize=(18, 14), max_nodes=200, seed=42):
    """
    Visualize the full knowledge graph as a network diagram.
    Uses spring layout and color-codes nodes by entity type.
    """
    print(f"\nBuilding network visualization...")

    G, entity_types, labels = build_networkx_from_rdf(rdf_path)
    print(f"  Total graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # If too large, take connected subgraph of most-connected nodes
    if G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda n: -degrees[n])[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        print(f"  Subsampled to {G.number_of_nodes()} highest-degree nodes")

    # Layout
    print("  Computing layout (spring)...")
    pos = nx.spring_layout(G, k=0.4, iterations=80, seed=seed)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.15, edge_color='#6B7280',
        width=0.5, arrows=False
    )

    # Draw nodes by type
    for etype, color in TYPE_COLORS.items():
        nodes = [n for n in G.nodes() if entity_types.get(n) == etype]
        if not nodes:
            continue

        # Size nodes by degree
        sizes = [TYPE_SIZES[etype] * (1 + 0.1 * G.degree(n)) for n in nodes]

        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_color=color, node_size=sizes,
            alpha=0.85, edgecolors='white', linewidths=1.0,
            label=f"{etype} ({len(nodes)})"
        )

    # Label top-degree publications only (avoid clutter)
    degrees = dict(G.degree())
    top_pubs = sorted(
        [n for n in G.nodes() if entity_types.get(n) == "Publication"],
        key=lambda n: -degrees.get(n, 0)
    )[:8]

    label_dict = {}
    for n in top_pubs:
        lbl = labels.get(n, "")
        if lbl:
            label_dict[n] = lbl[:35] + ("..." if len(lbl) > 35 else "")

    nx.draw_networkx_labels(
        G, pos, labels=label_dict, ax=ax,
        font_size=8, font_weight='bold',
        font_color='#1A1A2E',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#6B7280", alpha=0.85)
    )

    ax.set_title(
        "Machine Learning Research Knowledge Graph",
        fontsize=18, fontweight='bold', pad=20, color='#1A1A2E'
    )

    ax.text(
        0.5, -0.03,
        f"{G.number_of_nodes()} entities  ·  {G.number_of_edges()} relationships  ·  "
        f"{len(set(entity_types.values()))} entity types",
        transform=ax.transAxes, ha='center', fontsize=11,
        color='#6B7280', style='italic'
    )

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='#6B7280')
    ax.axis('off')

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Saved network visualization to {output_path}")


def visualize_citation_network(rdf_path, output_path="output/citation_network.png",
                                figsize=(16, 12), seed=42):
    """
    Focused visualization: just the citation network between publications.
    Node size by in-degree (papers cited most often in the KG).
    """
    print(f"\nBuilding citation network visualization...")

    G, entity_types, labels = build_networkx_from_rdf(rdf_path)

    # Filter to citation edges between publications only
    citation_graph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if (data.get('relation') == 'cites' and
                entity_types.get(u) == "Publication" and
                entity_types.get(v) == "Publication"):
            citation_graph.add_edge(u, v)

    if citation_graph.number_of_edges() == 0:
        print("  No citation edges found in graph")
        return

    print(f"  Citation network: {citation_graph.number_of_nodes()} papers, "
          f"{citation_graph.number_of_edges()} citations")

    # Layout
    pos = nx.spring_layout(citation_graph, k=1.0, iterations=100, seed=seed)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Size nodes by in-degree (most-cited = biggest)
    in_degrees = dict(citation_graph.in_degree())
    max_indeg = max(in_degrees.values()) if in_degrees else 1

    node_sizes = [200 + (in_degrees.get(n, 0) / max_indeg) * 2000
                  for n in citation_graph.nodes()]
    node_colors = ['#D97706' if in_degrees.get(n, 0) >= 3 else '#2E5E8A'
                   for n in citation_graph.nodes()]

    nx.draw_networkx_edges(
        citation_graph, pos, ax=ax,
        alpha=0.4, edge_color='#6B7280',
        arrows=True, arrowsize=12, arrowstyle='->',
        width=0.8, connectionstyle="arc3,rad=0.1"
    )

    nx.draw_networkx_nodes(
        citation_graph, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.85, edgecolors='white', linewidths=1.5
    )

    # Label most-cited papers
    top_cited = sorted(in_degrees.keys(),
                       key=lambda n: -in_degrees[n])[:10]
    label_dict = {}
    for n in top_cited:
        if in_degrees[n] >= 2:
            lbl = labels.get(n, "")
            if lbl:
                label_dict[n] = lbl[:32] + ("..." if len(lbl) > 32 else "")

    nx.draw_networkx_labels(
        citation_graph, pos, labels=label_dict, ax=ax,
        font_size=9, font_weight='bold', font_color='#1A1A2E',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#6B7280", alpha=0.9)
    )

    ax.set_title(
        "Citation Network: Foundational Papers in ML Research",
        fontsize=16, fontweight='bold', pad=20, color='#1A1A2E'
    )

    legend_elements = [
        mpatches.Patch(color='#D97706', label='Highly cited (3+ in-graph citations)'),
        mpatches.Patch(color='#2E5E8A', label='Other papers'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=10, framealpha=0.95)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Saved citation network to {output_path}")


def visualize_schema(output_path="output/schema_diagram.png",
                      figsize=(14, 10)):
    """
    Render a clean schema diagram showing the 8 classes and their relationships.
    This visualizes the ontology structure itself.
    """
    print(f"\nBuilding schema diagram...")

    G = nx.DiGraph()

    # Add classes as nodes
    classes = [
        "Publication", "Author", "Institution", "Venue",
        "ResearchArea", "ResearchTopic", "Dataset", "CodeRepository"
    ]
    for c in classes:
        G.add_node(c)

    # Add relationships
    relations = [
        ("Author", "Publication", "authorOf"),
        ("Publication", "Author", "firstAuthor"),
        ("Author", "Institution", "affiliatedWith"),
        ("Author", "Author", "coauthorWith"),
        ("Publication", "Publication", "cites"),
        ("Publication", "Venue", "publishedIn"),
        ("Publication", "ResearchTopic", "hasKeyword"),
        ("Publication", "ResearchArea", "inArea"),
        ("ResearchTopic", "ResearchArea", "topicInArea"),
        ("Publication", "Dataset", "usesDataset"),
        ("Publication", "CodeRepository", "hasCode"),
        ("CodeRepository", "Publication", "implementationOf"),
    ]

    for src, dst, rel in relations:
        G.add_edge(src, dst, relation=rel)

    # Manual positioning for clarity
    pos = {
        "Publication": (0, 0),
        "Author": (-3, 1),
        "Institution": (-5, 2),
        "Venue": (2, 2),
        "ResearchArea": (4, 0),
        "ResearchTopic": (2.5, -1.5),
        "Dataset": (-2, -2),
        "CodeRepository": (0, -2.5),
    }

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Draw edges with labels
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#6B7280', alpha=0.6,
        arrows=True, arrowsize=20, arrowstyle='->',
        width=1.5, connectionstyle="arc3,rad=0.1",
        node_size=4500
    )

    # Edge labels
    edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=8, font_color='#2E5E8A',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="none", alpha=0.9)
    )

    # Draw nodes
    for node in G.nodes():
        color = TYPE_COLORS.get(node, "#6B7280")
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], ax=ax,
            node_color=color, node_size=4500,
            alpha=0.9, edgecolors='white', linewidths=2
        )

    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=11, font_weight='bold', font_color='white'
    )

    ax.set_title(
        "Knowledge Graph Schema: 8 Classes, 12 Object Properties",
        fontsize=16, fontweight='bold', pad=20, color='#1A1A2E'
    )
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Saved schema diagram to {output_path}")


def visualize_entity_distribution(rdf_path,
                                    output_path="output/entity_distribution.png",
                                    figsize=(10, 6)):
    """Bar chart showing number of entities of each type in the KG."""
    print(f"\nBuilding entity distribution chart...")

    rdf_g = RDFGraph()
    rdf_g.parse(rdf_path, format="turtle")

    class_map = {
        "Publication": MLKG.Publication,
        "Author": MLKG.Author,
        "Institution": MLKG.Institution,
        "Venue": MLKG.Venue,
        "ResearchArea": MLKG.ResearchArea,
        "ResearchTopic": MLKG.ResearchTopic,
        "Dataset": MLKG.Dataset,
        "CodeRepository": MLKG.CodeRepository,
    }

    counts = {}
    for name, cls in class_map.items():
        counts[name] = len(list(rdf_g.subjects(RDF.type, cls)))

    # Filter out empty classes
    counts = {k: v for k, v in counts.items() if v > 0}

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    names = list(counts.keys())
    values = list(counts.values())
    colors = [TYPE_COLORS[n] for n in names]

    bars = ax.bar(names, values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                str(val), ha='center', fontsize=11, fontweight='bold',
                color='#1A1A2E')

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Knowledge Graph Entity Distribution",
                 fontsize=15, fontweight='bold', pad=15, color='#1A1A2E')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=30)
    plt.setp(ax.get_xticklabels(), ha='right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Saved entity distribution to {output_path}")


def visualize_all(rdf_path, output_dir="output"):
    """Generate all poster-ready visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GENERATING ALL VISUALIZATIONS")
    print("=" * 60)

    # 1. Schema diagram (for poster Fig 1)
    visualize_schema(str(output_dir / "schema_diagram.png"))

    # 2. Full KG network (for poster Graphical Abstract)
    visualize_full_graph(rdf_path, str(output_dir / "kg_network.png"),
                          max_nodes=150)

    # 3. Citation network (supplementary)
    visualize_citation_network(rdf_path, str(output_dir / "citation_network.png"))

    # 4. Entity distribution bar chart
    visualize_entity_distribution(rdf_path,
                                    str(output_dir / "entity_distribution.png"))

    print("\n" + "=" * 60)
    print("  ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"\nFiles saved in {output_dir}/:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    visualize_all(
        rdf_path="data/processed/ml_research_kg.ttl",
        output_dir="output"
    )
