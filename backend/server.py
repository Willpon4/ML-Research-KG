"""
ML Research KG — API Server
FastAPI backend serving the knowledge graph data.

Usage:
    pip install fastapi uvicorn rdflib
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from rdflib import Graph, Namespace, RDF, RDFS

# ── Config ──────────────────────────────────────────────────

KG_PATH = Path(__file__).parent.parent / "data" / "processed" / "ml_research_kg.ttl"
EMBEDDINGS_PATH = Path(__file__).parent.parent / "output" / "entity_embeddings.json"

MLKG = Namespace("http://example.org/mlkg/")

app = FastAPI(title="ML Research Knowledge Graph API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load KG at startup ──────────────────────────────────────

g = Graph()
entity_labels = {}
entity_types = {}
adjacency = defaultdict(list)

CLASS_MAP = {
    str(MLKG.Publication): "Publication",
    str(MLKG.Author): "Author",
    str(MLKG.Institution): "Institution",
    str(MLKG.Venue): "Venue",
    str(MLKG.ResearchArea): "ResearchArea",
    str(MLKG.ResearchTopic): "ResearchTopic",
    str(MLKG.Dataset): "Dataset",
    str(MLKG.CodeRepository): "CodeRepository",
}

SKIP_PREDS = {
    str(MLKG.title), str(MLKG.abstract),
    str(MLKG.publicationYear), str(MLKG.citationCount),
    str(RDFS.label), str(RDFS.comment),
    str(RDF.type), str(RDFS.domain), str(RDFS.range),
}


@app.on_event("startup")
def load_kg():
    global g, entity_labels, entity_types, adjacency

    print(f"Loading KG from {KG_PATH}...")
    g.parse(str(KG_PATH), format="turtle")
    print(f"  Loaded {len(g)} triples")

    # Build lookups
    for s, _, o in g.triples((None, RDFS.label, None)):
        entity_labels[str(s)] = str(o)

    for s, _, o in g.triples((None, RDF.type, None)):
        if str(o) in CLASS_MAP:
            entity_types[str(s)] = CLASS_MAP[str(o)]

    # Build adjacency
    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if p_str in SKIP_PREDS:
            continue
        if "www.w3.org" in s_str or "www.w3.org" in o_str:
            continue
        if not o_str.startswith("http"):
            continue
        if s_str not in entity_types or o_str not in entity_types:
            continue
        rel = p_str.split("/")[-1]
        adjacency[s_str].append({"target": o_str, "relation": rel})
        adjacency[o_str].append({"target": s_str, "relation": rel})

    print(f"  {len(entity_types)} entities, {len(adjacency)} with connections")


def node_to_dict(uri):
    """Convert an entity URI to a JSON-friendly dict."""
    return {
        "id": uri,
        "label": entity_labels.get(uri, uri.split("/")[-1]),
        "type": entity_types.get(uri, "Other"),
        "degree": len(adjacency.get(uri, [])),
    }


def get_node_details(uri):
    """Get full details about a node including its properties."""
    info = node_to_dict(uri)
    info["properties"] = {}

    # Get all literal properties
    for s, p, o in g.triples((None, None, None)):
        if str(s) == uri and not str(o).startswith("http"):
            prop_name = str(p).split("/")[-1]
            if prop_name not in ("label", "comment"):
                val = str(o)
                # Hide citationCount when 0 (arXiv doesn't provide it)
                if prop_name == "citationCount" and val == "0":
                    continue
                info["properties"][prop_name] = val

    return info


# ── Endpoints ───────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    """KG overview statistics."""
    type_counts = Counter(entity_types.values())
    rel_counts = Counter()
    for uri, neighbors in adjacency.items():
        for nb in neighbors:
            rel_counts[nb["relation"]] += 1
    # Divide by 2 since adjacency is bidirectional
    rel_counts = {k: v // 2 for k, v in rel_counts.items()}

    return {
        "total_triples": len(g),
        "total_entities": len(entity_types),
        "entity_counts": dict(type_counts),
        "relationship_counts": dict(rel_counts),
    }


@app.get("/api/graph")
def get_graph(
    exclude_rels: str = Query("", description="Comma-separated relations to exclude, e.g. 'coauthorWith'"),
    exclude_types: str = Query("", description="Comma-separated entity types to exclude"),
):
    """
    Get the full graph for visualization.
    Use exclude_rels to hide noisy edges (e.g. coauthorWith).
    Use exclude_types to hide entity types (e.g. Author).
    """
    skip_rels = set(r.strip() for r in exclude_rels.split(",") if r.strip())
    skip_types = set(t.strip() for t in exclude_types.split(",") if t.strip())

    # All nodes (optionally filtered by type)
    selected = set()
    for uri, etype in entity_types.items():
        if etype not in skip_types:
            selected.add(uri)

    nodes = [node_to_dict(uri) for uri in selected]

    # All edges (optionally filtered by relation)
    edges = []
    seen_edges = set()
    for uri in selected:
        for nb in adjacency.get(uri, []):
            if nb["target"] in selected and nb["relation"] not in skip_rels:
                edge_key = tuple(sorted([uri, nb["target"]])) + (nb["relation"],)
                if edge_key not in seen_edges:
                    edges.append({
                        "source": uri,
                        "target": nb["target"],
                        "relation": nb["relation"],
                    })
                    seen_edges.add(edge_key)

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


@app.get("/api/node/{node_id:path}")
def get_node(node_id: str):
    """Get details about a specific node."""
    if node_id not in entity_types:
        return {"error": "Node not found"}
    return get_node_details(node_id)


@app.get("/api/neighbors/{node_id:path}")
def get_neighbors(node_id: str, limit: int = Query(50, le=200)):
    """Get a node and its immediate neighbors as a subgraph."""
    if node_id not in entity_types:
        return {"error": "Node not found"}

    neighbors = adjacency.get(node_id, [])[:limit]
    selected = {node_id}
    edges = []

    for nb in neighbors:
        selected.add(nb["target"])
        edges.append({
            "source": node_id,
            "target": nb["target"],
            "relation": nb["relation"],
        })

    nodes = [node_to_dict(uri) for uri in selected]
    return {"nodes": nodes, "edges": edges, "center": node_id}


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    type: str = Query(None),
    limit: int = Query(20, le=100),
):
    """Search entities by name."""
    q_lower = q.lower()
    results = []

    for uri, label in entity_labels.items():
        if q_lower in label.lower():
            etype = entity_types.get(uri)
            if type and etype != type:
                continue
            results.append(node_to_dict(uri))
            if len(results) >= limit:
                break

    # Sort by degree (most connected first)
    results.sort(key=lambda x: -x["degree"])
    return {"results": results, "total": len(results)}


# ── SPARQL Queries ──────────────────────────────────────────

USE_CASES = {
    "recommendation": {
        "title": "Paper Recommendation",
        "description": "Find papers that share research topics with a target paper.",
        "question": "What should I read next?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?title (COUNT(?sharedTopic) AS ?topicOverlap) ?year WHERE {
    ?target mlkg:title ?targetTitle .
    FILTER(CONTAINS(LCASE(?targetTitle), LCASE("PLACEHOLDER")))
    ?target mlkg:hasKeyword ?sharedTopic .
    ?paper mlkg:hasKeyword ?sharedTopic .
    ?paper mlkg:title ?title .
    OPTIONAL { ?paper mlkg:publicationYear ?year }
    FILTER(?paper != ?target)
}
GROUP BY ?paper ?title ?year
ORDER BY DESC(?topicOverlap)
LIMIT 10
""",
        "default_param": "Attention",
        "param_label": "Paper title (or keyword)",
    },

    "trends": {
        "title": "Emerging Trend Detection",
        "description": "Identify research topics with the most publications since 2020.",
        "question": "What's hot right now?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?topicName (COUNT(?paper) AS ?paperCount) WHERE {
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label ?topicName .
    ?paper mlkg:publicationYear ?year .
    FILTER(?year >= 2020)
}
GROUP BY ?topic ?topicName
ORDER BY DESC(?paperCount)
LIMIT 15
""",
        "default_param": None,
        "param_label": None,
    },

    "foundational": {
        "title": "Foundational Papers",
        "description": "Find key papers on a specific research topic.",
        "question": "What are the key papers on this topic?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?title ?year WHERE {
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label ?topicLabel .
    FILTER(CONTAINS(LCASE(?topicLabel), LCASE("PLACEHOLDER")))
    ?paper mlkg:title ?title .
    OPTIONAL { ?paper mlkg:publicationYear ?year }
}
ORDER BY ?year
LIMIT 15
""",
        "default_param": "Transformer",
        "param_label": "Topic keyword",
    },

    "experts": {
        "title": "Expert Discovery",
        "description": "Find the most prolific authors in a research area.",
        "question": "Who are the leading researchers?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?authorName (COUNT(DISTINCT ?paper) AS ?paperCount) WHERE {
    ?author mlkg:authorOf ?paper .
    ?author rdfs:label ?authorName .
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label ?topicLabel .
    FILTER(CONTAINS(LCASE(?topicLabel), LCASE("PLACEHOLDER")))
}
GROUP BY ?author ?authorName
ORDER BY DESC(?paperCount)
LIMIT 15
""",
        "default_param": "Graph Neural",
        "param_label": "Topic keyword",
    },

    "timeline": {
        "title": "Research Timeline",
        "description": "Track how publication volume has changed year by year.",
        "question": "How has this field evolved?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?year (COUNT(?paper) AS ?paperCount) WHERE {
    ?paper mlkg:publicationYear ?year .
    FILTER(?year >= 2015 && ?year <= 2025)
}
GROUP BY ?year
ORDER BY ?year
""",
        "default_param": None,
        "param_label": None,
    },

    "collaboration": {
        "title": "Collaboration Discovery",
        "description": "Find the most frequent co-author pairs in the knowledge graph.",
        "question": "Who works together?",
        "query": """
PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?author1Name ?author2Name (COUNT(?paper) AS ?sharedPapers) WHERE {
    ?author1 mlkg:coauthorWith ?author2 .
    ?author1 rdfs:label ?author1Name .
    ?author2 rdfs:label ?author2Name .
    ?author1 mlkg:authorOf ?paper .
    ?author2 mlkg:authorOf ?paper .
    FILTER(STR(?author1) < STR(?author2))
}
GROUP BY ?author1Name ?author2Name
ORDER BY DESC(?sharedPapers)
LIMIT 15
""",
        "default_param": None,
        "param_label": None,
    },
}


@app.get("/api/queries")
def list_queries():
    """List all available use case queries."""
    return {
        key: {
            "title": uc["title"],
            "description": uc["description"],
            "question": uc["question"],
            "param_label": uc["param_label"],
            "default_param": uc["default_param"],
        }
        for key, uc in USE_CASES.items()
    }


@app.get("/api/query/{use_case_id}")
def run_query(use_case_id: str, param: str = Query(None)):
    """Run a SPARQL query for a use case and return results."""
    if use_case_id not in USE_CASES:
        return {"error": f"Unknown use case: {use_case_id}"}

    uc = USE_CASES[use_case_id]
    query = uc["query"]

    # Substitute parameter if needed
    if "PLACEHOLDER" in query:
        value = param or uc["default_param"] or ""
        query = query.replace("PLACEHOLDER", value)

    # Show the query text to the user
    display_query = query.strip()

    try:
        results = g.query(query)
        rows = []
        columns = [str(v) for v in results.vars] if results.vars else []

        for row in results:
            row_dict = {}
            for i, var in enumerate(results.vars):
                val = row[i]
                if val is not None:
                    val_str = str(val)
                    # Clean up URIs for display
                    if val_str.startswith("http://example.org/mlkg/data/"):
                        val_str = val_str.split("/")[-1]
                    row_dict[str(var)] = val_str
                else:
                    row_dict[str(var)] = None
            rows.append(row_dict)

        return {
            "use_case": uc["title"],
            "description": uc["description"],
            "query": display_query,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "query": display_query}


# ── Embeddings ──────────────────────────────────────────────

@app.get("/api/embeddings")
def get_embeddings(limit: int = Query(500, le=1000)):
    """Get entity embeddings for visualization (pre-reduced to 2D)."""
    if not EMBEDDINGS_PATH.exists():
        return {"error": "Embeddings not found. Run the ML pipeline first."}

    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    with open(EMBEDDINGS_PATH) as f:
        raw = json.load(f)

    # Sample proportionally by type
    by_type = defaultdict(list)
    for entity, emb in raw.items():
        etype = entity_types.get(entity, "Other")
        by_type[etype].append((entity, emb))

    max_per_type = {"Publication": 150, "Author": 200}
    sampled = []
    rng = np.random.RandomState(42)

    for etype, items in by_type.items():
        max_n = max_per_type.get(etype)
        if max_n and len(items) > max_n:
            indices = rng.choice(len(items), max_n, replace=False)
            items = [items[i] for i in indices]
        for entity, emb in items:
            sampled.append({"entity": entity, "emb": emb,
                            "type": etype,
                            "label": entity_labels.get(entity, entity.split("/")[-1])})

    if not sampled:
        return {"error": "No embeddings found"}

    X = np.array([s["emb"] for s in sampled])

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1),
                random_state=42, max_iter=1000, init='pca')
    X_tsne = tsne.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    points = []
    for i, s in enumerate(sampled):
        points.append({
            "label": s["label"][:50],
            "type": s["type"],
            "tsne": {"x": float(X_tsne[i, 0]), "y": float(X_tsne[i, 1])},
            "pca": {"x": float(X_pca[i, 0]), "y": float(X_pca[i, 1])},
        })

    return {"points": points, "total": len(points)}