# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS 497 Knowledge Graph Engineering course project. Builds a semantic RDF knowledge graph of ML/AI research papers, queryable via SPARQL, with graph embeddings for link prediction and a React+FastAPI web frontend.

## Commands

### Python Pipeline
```bash
pip install -r requirements.txt          # install core deps (PyKEEN optional for advanced embeddings)
python scripts/main.py                   # full pipeline: extract → build KG → SPARQL → embed → visualize
python scripts/main.py --skip-extraction # skip API calls, rebuild from cached data/raw/ JSON
python scripts/main.py --queries-only    # only run SPARQL queries on existing KG
```

### Backend
```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend && npm install
npm run dev      # dev server
npm run build    # production build
npm run lint     # ESLint
```

### Visualization Scripts (standalone)
```bash
python scripts/viz_kg_interactive.py        # interactive KG viewer
python scripts/viz_kg_subgraph.py           # subgraph analysis
python scripts/viz_embeddings.py            # t-SNE/PCA plots
python scripts/viz_embeddings_interactive.py
```

## Architecture

### Data Flow
```
APIs (Semantic Scholar, arXiv, Papers with Code)
  → src/extractors/     raw JSON → data/raw/
  → src/extractors/merger.py    consolidated JSON
  → src/kg/builder.py   RDF triples → data/processed/*.ttl
  → src/queries/sparql_queries.py   SPARQL over RDF graph
  → src/ml/embeddings.py            entity embeddings + link prediction
  → src/visualization/              matplotlib/D3 output
  → backend/server.py               FastAPI serving the KG
  → frontend/                       React+D3 interactive viewer
```

### Key Classes
- **`src/kg/builder.py` — `KnowledgeGraphBuilder`**: Core triple generator. Handles 8 entity types and 16 relationship types. Multi-phase construction: publications → authors → venues → citations → topics → code repos.
- **`src/ml/embeddings.py` — `KGEmbeddingPipeline`**: TransE/RotatE/ComplEx via PyKEEN; falls back to SVD if PyKEEN not installed.
- **`src/extractors/sample_data.py`**: Generates 35 landmark ML papers with realistic metadata — use this for development without hitting APIs.
- **`scripts/main.py`**: Orchestrates all 6 phases; flags `--skip-extraction` and `--queries-only` control which phases run.
- **`backend/server.py`**: FastAPI app that loads the `.ttl` graph at startup, builds entity/adjacency indexes, exposes REST endpoints for the frontend.

### Schema
Ontology defined in `schema/ontology.ttl` and `schema/ml_research_ontology.ttl`. Namespace prefix: `mlkg:`. Entity types: Author, Publication, Institution, Venue, ResearchArea, ResearchTopic, Dataset, CodeRepository.

### SPARQL Queries (`src/queries/sparql_queries.py`)
10 predefined queries covering: paper recommendations, emerging trend detection, foundational papers, expert discovery, research timeline, collaboration networks.

## Configuration
- **Semantic Scholar API key**: Optional; set `SEMANTIC_SCHOLAR_API_KEY` env var to avoid rate limiting.
- **No database**: The KG lives in-memory (RDFLib) loaded from `.ttl` files in `data/processed/`.
- **PyKEEN**: Optional. If not installed, embeddings fall back to SVD-based approach.
