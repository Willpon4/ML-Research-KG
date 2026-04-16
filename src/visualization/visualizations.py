"""
Extended visualizations for the ML Research KG.
Generates all the visualizations needed for poster and reports:

1. Schema diagram (ontology classes and relationships)
2. Knowledge graph subgraph (NetworkX visualization)
3. Data pipeline diagram
4. Research timeline chart
5. Topic/area distribution
6. Citation network visualization
7. Co-authorship network
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import networkx as nx

from rdflib import Graph, Namespace, RDF, RDFS

MLKG = Namespace("http://example.org/mlkg/")

# Color palette (matches poster)
COLORS = {
    "Publication": "#2E5E8A",
    "Author": "#D97706",
    "Institution": "#059669",
    "Venue": "#DC2626",
    "ResearchArea": "#7C3AED",
    "ResearchTopic": "#DB2777",
    "Dataset": "#0891B2",
    "CodeRepository": "#65A30D",
}

# Poster palette
POSTER_BG = "#FAFAF7"
POSTER_TEXT = "#1A1A2E"
POSTER_BLUE = "#2E5E8A"
POSTER_AMBER = "#D97706"
POSTER_GRAY = "#6B7280"


class KGVisualizer:
    """Generate visualizations for the ML Research KG."""

    def __init__(self, kg_path, output_dir="output"):
        self.kg_path = kg_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.g = Graph()
        self.g.parse(kg_path, format="turtle")
        print(f"Loaded graph with {len(self.g)} triples")

        # Build entity lookup
        self.entity_labels = {}
        self.entity_types = {}
        for s, _, o in self.g.triples((None, RDFS.label, None)):
            self.entity_labels[str(s)] = str(o)

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
        for s, _, o in self.g.triples((None, RDF.type, None)):
            if str(o) in class_map:
                self.entity_types[str(s)] = class_map[str(o)]

    # ============================================================
    # 1. Schema Diagram
    # ============================================================

    def draw_schema_diagram(self, output_file="fig1_schema.png"):
        """Draw the ontology schema: 8 classes and their relationships."""
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=POSTER_BG)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Class positions (x, y, width, height)
        classes = {
            "Publication":    (6, 5, 2.2, 1.0),
            "Author":         (1.5, 7.5, 1.8, 0.9),
            "Institution":    (1, 5, 1.8, 0.9),
            "Venue":          (11, 7.5, 1.5, 0.9),
            "ResearchArea":   (11, 5, 1.8, 0.9),
            "ResearchTopic": (9.5, 2.5, 1.8, 0.9),
            "Dataset":        (3, 2.5, 1.5, 0.9),
            "CodeRepository": (6, 2.5, 2.0, 0.9),
        }

        # Draw classes as rounded rectangles
        for name, (x, y, w, h) in classes.items():
            color = COLORS[name]
            box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white',
                linewidth=2, alpha=0.9
            )
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, name,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')

        # Relationships: (from_class, to_class, label, curve)
        relationships = [
            ("Author", "Publication", "authorOf", 0),
            ("Author", "Institution", "affiliatedWith", 0.2),
            ("Publication", "Author", "firstAuthor", 0.3),
            ("Publication", "Publication", "cites", -0.6),
            ("Publication", "Venue", "publishedIn", 0),
            ("Publication", "ResearchArea", "inArea", 0),
            ("Publication", "ResearchTopic", "hasKeyword", 0),
            ("ResearchTopic", "ResearchArea", "topicInArea", 0),
            ("Publication", "Dataset", "usesDataset", 0),
            ("Publication", "CodeRepository", "hasCode", 0),
        ]

        for from_c, to_c, label, curve in relationships:
            fx, fy, fw, fh = classes[from_c]
            tx, ty, tw, th = classes[to_c]

            # Center points
            from_pt = (fx + fw/2, fy + fh/2)
            to_pt = (tx + tw/2, ty + th/2)

            # Self-loop for cites
            if from_c == to_c:
                loop = mpatches.FancyArrowPatch(
                    (fx + fw, fy + fh - 0.1),
                    (fx + fw, fy + 0.1),
                    connectionstyle=f"arc3,rad={curve}",
                    arrowstyle='->,head_width=0.3,head_length=0.4',
                    color=POSTER_GRAY, linewidth=1.5
                )
                ax.add_patch(loop)
                ax.text(fx + fw + 0.8, fy + fh/2, label,
                        fontsize=8, style='italic', color=POSTER_TEXT,
                        ha='left', va='center')
            else:
                arrow = FancyArrowPatch(
                    from_pt, to_pt,
                    connectionstyle=f"arc3,rad={curve}",
                    arrowstyle='->,head_width=0.3,head_length=0.4',
                    color=POSTER_GRAY, linewidth=1.5,
                    shrinkA=35, shrinkB=35
                )
                ax.add_patch(arrow)

                # Label position
                mid_x = (from_pt[0] + to_pt[0]) / 2
                mid_y = (from_pt[1] + to_pt[1]) / 2
                # Offset for curve
                if curve != 0:
                    mid_y += curve * 0.5
                ax.text(mid_x, mid_y, label,
                        fontsize=8, style='italic', color=POSTER_TEXT,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=POSTER_BG,
                                  edgecolor='none', alpha=0.8))

        # Title
        ax.text(7, 9.5, "ML Research KG Schema",
                ha='center', fontsize=16, fontweight='bold',
                color=POSTER_BLUE)
        ax.text(7, 9.1, "8 Classes · 12 Object Properties · 4 Datatype Properties",
                ha='center', fontsize=10, color=POSTER_GRAY)

        # Datatype properties note
        ax.text(7, 0.5,
                "Datatype properties on Publication: title · abstract · publicationYear · citationCount",
                ha='center', fontsize=9, style='italic', color=POSTER_GRAY)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved schema diagram to {output_path}")

    # ============================================================
    # 2. Data Pipeline Diagram
    # ============================================================

    def draw_pipeline_diagram(self, output_file="fig2_pipeline.png"):
        """Draw the data extraction and KG construction pipeline."""
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=POSTER_BG)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Data sources
        sources = [
            (0.5, 4.2, "Semantic Scholar", "#2E5E8A"),
            (0.5, 2.8, "arXiv", "#7C3AED"),
            (0.5, 1.4, "Papers with Code", "#059669"),
        ]

        for x, y, label, color in sources:
            box = FancyBboxPatch(
                (x, y), 2.3, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white',
                linewidth=2, alpha=0.9
            )
            ax.add_patch(box)
            ax.text(x + 1.15, y + 0.45, label,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold')

        # Processing steps
        steps = [
            (4, 4, "Extraction\n(API calls)", POSTER_AMBER),
            (7, 4, "Entity\nResolution", POSTER_AMBER),
            (10, 4, "RDF Triple\nGeneration", POSTER_AMBER),
        ]

        for x, y, label, color in steps:
            box = FancyBboxPatch(
                (x, y - 0.5), 2.2, 1.5,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='white',
                linewidth=2, alpha=0.9
            )
            ax.add_patch(box)
            ax.text(x + 1.1, y + 0.25, label,
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

        # Final KG box
        kg_box = FancyBboxPatch(
            (12.3, 1.8), 1.6, 2.8,
            boxstyle="round,pad=0.05",
            facecolor=POSTER_BLUE, edgecolor='white',
            linewidth=2.5, alpha=0.95
        )
        ax.add_patch(kg_box)
        ax.text(13.1, 3.8, "Knowledge\nGraph",
                ha='center', va='center', color='white',
                fontsize=12, fontweight='bold')
        ax.text(13.1, 2.8, f"{len(self.g)}\ntriples",
                ha='center', va='center', color='white', fontsize=10)

        # Arrows from sources to extraction
        for _, sy, _, _ in sources:
            arrow = FancyArrowPatch(
                (2.8, sy + 0.45), (4, 4.4),
                arrowstyle='->,head_width=0.25,head_length=0.35',
                color=POSTER_GRAY, linewidth=1.5,
                connectionstyle="arc3,rad=0.1"
            )
            ax.add_patch(arrow)

        # Arrows between steps
        for i in range(len(steps) - 1):
            x1 = steps[i][0] + 2.2
            x2 = steps[i + 1][0]
            y = steps[i][1] + 0.25
            arrow = FancyArrowPatch(
                (x1, y), (x2, y),
                arrowstyle='->,head_width=0.3,head_length=0.4',
                color=POSTER_GRAY, linewidth=2
            )
            ax.add_patch(arrow)

        # Arrow from last step to KG
        arrow = FancyArrowPatch(
            (12.2, 4.25), (12.3, 3.2),
            arrowstyle='->,head_width=0.3,head_length=0.4',
            color=POSTER_GRAY, linewidth=2
        )
        ax.add_patch(arrow)

        # Counts of entities below
        counts = self._get_entity_counts()
        total_entities = sum(counts.values())
        ax.text(7, 0.5,
                f"Extracted: {counts.get('Publication', 0)} papers · "
                f"{counts.get('Author', 0)} authors · "
                f"{counts.get('Institution', 0)} institutions · "
                f"{counts.get('Venue', 0)} venues",
                ha='center', fontsize=10, color=POSTER_TEXT,
                fontweight='bold')

        # Title
        ax.text(7, 5.5, "Data Extraction Pipeline",
                ha='center', fontsize=16, fontweight='bold',
                color=POSTER_BLUE)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved pipeline diagram to {output_path}")

    def _get_entity_counts(self):
        """Count entities by type."""
        counts = Counter()
        for etype in self.entity_types.values():
            counts[etype] += 1
        return counts

    # ============================================================
    # 3. KG Subgraph Visualization (NetworkX)
    # ============================================================

    def draw_kg_subgraph(self, output_file="graphical_abstract.png",
                          max_nodes=80):
        """
        Draw a subgraph of the KG showing a representative slice.
        Used as the 'Graphical Abstract' on the poster.
        """
        # Build NetworkX graph from RDF
        G = nx.Graph()

        # Gather object property triples (skip literals)
        skip_preds = {str(MLKG.title), str(MLKG.abstract),
                       str(MLKG.publicationYear), str(MLKG.citationCount),
                       str(RDFS.label), str(RDFS.comment),
                       str(RDF.type), str(RDFS.domain), str(RDFS.range)}

        edges_added = []
        for s, p, o in self.g:
            s_str, p_str, o_str = str(s), str(p), str(o)
            if p_str in skip_preds:
                continue
            if "www.w3.org" in s_str or "www.w3.org" in o_str:
                continue
            if not o_str.startswith("http"):
                continue
            if s_str not in self.entity_types or o_str not in self.entity_types:
                continue

            G.add_edge(s_str, o_str, relation=p_str.split("/")[-1])
            edges_added.append((s_str, o_str))

        print(f"Full KG graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

        # Select subgraph: most connected publications + their neighbors
        pub_nodes = [n for n in G.nodes()
                     if self.entity_types.get(n) == "Publication"]

        if not pub_nodes:
            print("No publications found in graph")
            return

        # Sort by degree
        pub_degrees = [(n, G.degree(n)) for n in pub_nodes]
        pub_degrees.sort(key=lambda x: -x[1])

        # Take top publications and their neighbors
        seed_nodes = set([n for n, _ in pub_degrees[:15]])
        neighbors = set()
        for n in seed_nodes:
            neighbors.update(G.neighbors(n))

        subgraph_nodes = list(seed_nodes | neighbors)[:max_nodes]
        SG = G.subgraph(subgraph_nodes).copy()

        print(f"Subgraph: {SG.number_of_nodes()} nodes, "
              f"{SG.number_of_edges()} edges")

        # Layout
        pos = nx.spring_layout(SG, k=2.5, iterations=100, seed=42)

        # Draw
        fig, ax = plt.subplots(figsize=(14, 11), facecolor=POSTER_BG)

        # Draw edges
        nx.draw_networkx_edges(
            SG, pos, ax=ax,
            edge_color=POSTER_GRAY,
            width=0.6, alpha=0.35
        )

        # Draw nodes by type
        for etype, color in COLORS.items():
            nodes = [n for n in SG.nodes()
                     if self.entity_types.get(n) == etype]
            if not nodes:
                continue

            # Size by degree
            sizes = [80 + SG.degree(n) * 40 for n in nodes]

            nx.draw_networkx_nodes(
                SG, pos, nodelist=nodes,
                node_color=color, node_size=sizes,
                alpha=0.85, edgecolors='white', linewidths=1.5,
                ax=ax, label=f"{etype}"
            )

        # Label only top-degree nodes
        labels = {}
        top_nodes = sorted(SG.nodes(), key=lambda n: -SG.degree(n))[:20]
        for n in top_nodes:
            label = self.entity_labels.get(n, "")
            if label:
                # Truncate
                if len(label) > 28:
                    label = label[:25] + "..."
                labels[n] = label

        nx.draw_networkx_labels(
            SG, pos, labels=labels,
            font_size=7, font_color=POSTER_TEXT,
            font_weight='bold',
            ax=ax,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor=POSTER_BG, edgecolor='none', alpha=0.7)
        )

        # Legend
        ax.legend(
            loc='lower left', fontsize=10, frameon=True,
            facecolor=POSTER_BG, edgecolor=POSTER_GRAY,
            title="Entity Type", title_fontsize=11
        )

        # Title and stats
        total_counts = self._get_entity_counts()
        ax.set_title(
            f"ML Research Knowledge Graph",
            fontsize=18, fontweight='bold', color=POSTER_BLUE, pad=20
        )

        stats_text = (
            f"{len(self.g)} triples · "
            f"{total_counts.get('Publication', 0)} papers · "
            f"{total_counts.get('Author', 0)} authors · "
            f"{total_counts.get('Institution', 0)} institutions"
        )
        ax.text(0.5, -0.02, stats_text,
                transform=ax.transAxes, ha='center',
                fontsize=11, color=POSTER_TEXT, fontweight='bold')

        ax.axis('off')
        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved KG subgraph to {output_path}")

    # ============================================================
    # 4. Research Timeline
    # ============================================================

    def draw_timeline(self, output_file="fig_timeline.png"):
        """Chart showing papers and citations over time."""
        q = """
        PREFIX mlkg: <http://example.org/mlkg/>
        SELECT ?year (COUNT(?paper) AS ?papers) (SUM(?cit) AS ?totalCit)
        WHERE {
            ?paper a mlkg:Publication .
            ?paper mlkg:publicationYear ?year .
            ?paper mlkg:citationCount ?cit .
        } GROUP BY ?year ORDER BY ?year
        """

        years, papers, citations = [], [], []
        for row in self.g.query(q):
            years.append(int(row.year))
            papers.append(int(row.papers))
            citations.append(int(row.totalCit))

        if not years:
            print("No year data available")
            return

        fig, ax1 = plt.subplots(figsize=(11, 6), facecolor=POSTER_BG)
        ax1.set_facecolor(POSTER_BG)

        # Bar chart for paper count
        bars = ax1.bar(years, papers, color=POSTER_BLUE, alpha=0.75,
                        edgecolor='white', linewidth=1.5, label="Papers")
        ax1.set_xlabel("Publication Year", fontsize=12, fontweight='bold',
                        color=POSTER_TEXT)
        ax1.set_ylabel("Number of Papers", fontsize=12, fontweight='bold',
                        color=POSTER_BLUE)
        ax1.tick_params(axis='y', labelcolor=POSTER_BLUE)

        # Line chart for citations (secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(years, citations, color=POSTER_AMBER, marker='o',
                  linewidth=2.5, markersize=8, label="Total Citations")
        ax2.set_ylabel("Total Citations", fontsize=12, fontweight='bold',
                        color=POSTER_AMBER)
        ax2.tick_params(axis='y', labelcolor=POSTER_AMBER)

        # Value labels on bars
        for bar, p in zip(bars, papers):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(p), ha='center', fontsize=9, color=POSTER_TEXT)

        ax1.set_title("Research Timeline: Papers & Citations by Year",
                       fontsize=15, fontweight='bold', color=POSTER_BLUE, pad=15)

        # Grid
        ax1.grid(True, axis='y', alpha=0.25, linestyle='--')
        ax1.set_axisbelow(True)

        # Remove spines
        for spine in ['top']:
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved timeline to {output_path}")

    # ============================================================
    # 5. Topic Distribution / Emerging Trends
    # ============================================================

    def draw_topic_distribution(self, output_file="fig_topics.png"):
        """Bar chart of paper counts by research topic."""
        q = """
        PREFIX mlkg: <http://example.org/mlkg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?topicName (COUNT(?paper) AS ?count) (AVG(?cit) AS ?avgCit)
        WHERE {
            ?paper mlkg:hasKeyword ?topic .
            ?topic rdfs:label ?topicName .
            ?paper mlkg:citationCount ?cit .
        } GROUP BY ?topicName ORDER BY DESC(?count)
        """

        topics, counts, avg_cits = [], [], []
        for row in self.g.query(q):
            topics.append(str(row.topicName))
            counts.append(int(row['count']))
            avg_cits.append(float(row.avgCit))

        if not topics:
            print("No topic data")
            return

        # Take top 12
        topics = topics[:12]
        counts = counts[:12]
        avg_cits = avg_cits[:12]

        fig, ax = plt.subplots(figsize=(11, 7), facecolor=POSTER_BG)
        ax.set_facecolor(POSTER_BG)

        y_pos = np.arange(len(topics))

        # Bar colors based on avg citations
        max_cit = max(avg_cits) if avg_cits and max(avg_cits) > 0 else 1
        colors = [plt.cm.YlOrRd(0.3 + 0.6 * (c / max_cit)) for c in avg_cits]

        bars = ax.barh(y_pos, counts, color=colors, edgecolor='white',
                        linewidth=1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(topics, fontsize=10, color=POSTER_TEXT)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Papers", fontsize=12, fontweight='bold',
                       color=POSTER_TEXT)

        # Value labels
        for bar, count, avg in zip(bars, counts, avg_cits):
            w = bar.get_width()
            ax.text(w + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                     f"{count} papers  ·  avg {int(avg):,} cit",
                     va='center', fontsize=9, color=POSTER_TEXT)

        ax.set_title("Research Topics in the Knowledge Graph",
                      fontsize=15, fontweight='bold',
                      color=POSTER_BLUE, pad=15)

        ax.grid(True, axis='x', alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved topic distribution to {output_path}")

    # ============================================================
    # 6. Entity Distribution Pie / Summary Stats
    # ============================================================

    def draw_entity_breakdown(self, output_file="fig_entities.png"):
        """Pie chart showing entity type breakdown."""
        counts = self._get_entity_counts()
        if not counts:
            return

        labels = list(counts.keys())
        values = list(counts.values())
        colors = [COLORS.get(l, POSTER_GRAY) for l in labels]

        fig, ax = plt.subplots(figsize=(9, 7), facecolor=POSTER_BG)
        ax.set_facecolor(POSTER_BG)

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct=lambda p: f'{int(p*sum(values)/100)}\n({p:.0f}%)',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )

        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
            at.set_fontsize(9)

        for t in texts:
            t.set_fontsize(11)
            t.set_color(POSTER_TEXT)

        ax.set_title(f"Knowledge Graph Entity Breakdown\n{sum(values)} total entities",
                      fontsize=14, fontweight='bold', color=POSTER_BLUE, pad=20)
        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved entity breakdown to {output_path}")

    # ============================================================
    # 7. Co-authorship network
    # ============================================================

    def draw_coauthor_network(self, output_file="fig_coauthors.png",
                                min_papers=1, max_authors=40):
        """Visualize the co-authorship network."""
        # Find authors by paper count
        q = """
        PREFIX mlkg: <http://example.org/mlkg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?author ?name (COUNT(?paper) AS ?pc) WHERE {
            ?author a mlkg:Author .
            ?author rdfs:label ?name .
            ?author mlkg:authorOf ?paper .
        } GROUP BY ?author ?name ORDER BY DESC(?pc)
        """

        top_authors = []
        for row in self.g.query(q):
            top_authors.append((str(row.author), str(row['name']),
                                int(row.pc)))
            if len(top_authors) >= max_authors:
                break

        if not top_authors:
            print("No authors found")
            return

        top_author_uris = set([a[0] for a in top_authors])

        # Build co-authorship edges
        G = nx.Graph()
        for uri, name, pc in top_authors:
            G.add_node(uri, name=name, papers=pc)

        # Find co-authorships
        for s, _, o in self.g.triples((None, MLKG.coauthorWith, None)):
            if str(s) in top_author_uris and str(o) in top_author_uris:
                G.add_edge(str(s), str(o))

        # Remove isolated nodes
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)

        if G.number_of_nodes() == 0:
            print("No co-authorship network to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 10), facecolor=POSTER_BG)
        ax.set_facecolor(POSTER_BG)

        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

        # Edges
        nx.draw_networkx_edges(G, pos, ax=ax,
                                edge_color=POSTER_GRAY, alpha=0.4, width=0.8)

        # Nodes sized by paper count
        sizes = [200 + G.nodes[n].get('papers', 1) * 100 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=COLORS["Author"],
                                node_size=sizes, alpha=0.85,
                                edgecolors='white', linewidths=1.5, ax=ax)

        # Labels
        labels = {n: G.nodes[n]['name'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8,
                                 font_color=POSTER_TEXT, font_weight='bold',
                                 ax=ax,
                                 bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor=POSTER_BG,
                                           edgecolor='none', alpha=0.75))

        ax.set_title(f"Co-authorship Network ({G.number_of_nodes()} authors, "
                      f"{G.number_of_edges()} connections)",
                      fontsize=14, fontweight='bold',
                      color=POSTER_BLUE, pad=15)
        ax.axis('off')
        plt.tight_layout()
        output_path = self.output_dir / output_file
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=POSTER_BG)
        plt.close(fig)
        print(f"Saved co-authorship network to {output_path}")

    # ============================================================
    # Generate all visualizations
    # ============================================================

    def generate_all(self):
        """Generate the full suite of visualizations."""
        print("\n" + "=" * 60)
        print("  GENERATING ALL VISUALIZATIONS")
        print("=" * 60)

        self.draw_schema_diagram()
        self.draw_pipeline_diagram()
        self.draw_kg_subgraph()
        self.draw_timeline()
        self.draw_topic_distribution()
        self.draw_entity_breakdown()
        self.draw_coauthor_network()

        print("\nAll visualizations saved to", self.output_dir)


if __name__ == "__main__":
    viz = KGVisualizer("data/processed/ml_research_kg.ttl", "output")
    viz.generate_all()
