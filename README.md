# Machine Learning Research Knowledge Graph

A knowledge graph for exploring ML/AI research papers, enabling paper recommendations, trend detection, and research network analysis.

## Overview

This project builds a semantic knowledge graph that connects machine learning research papers, authors, institutions, citations, datasets, and code repositories. The system demonstrates knowledge graph engineering techniques including RDF/OWL schema design, SPARQL querying, graph embeddings, and an interactive web interface.

**Course:** CS 497 - Knowledge Graph Engineering (15 weeks)
**Technology Stack:** RDF/OWL, SPARQL, Python, FastAPI, React

---

## Use Cases

1. **Paper Recommendation:** "What should I read next?" - Find similar papers through citations and topics
2. **Emerging Trends:** "What's hot right now?" - Detect topics with growing citation counts
3. **Foundational Papers:** "What are the key papers on transformers?" - Identify highly-cited foundational works
4. **Expert Discovery:** "Who are the leading GNN researchers?" - Rank authors by impact
5. **Research Timeline:** "How did deep learning evolve?" - Trace citation chains over time
6. **Collaboration Discovery:** "Who works together?" - Map co-authorship networks

---

## Schema

The knowledge graph uses 8 entity types and 16 relationship types:

**Classes:** Author, Institution, Publication, Venue, ResearchArea, ResearchTopic, Dataset, CodeRepository

**Key Relationships:** authorOf, cites, publishedIn, hasKeyword, usesDataset, hasCode, coauthorWith

See [schema/ml_research_ontology.ttl](schema/ml_research_ontology.ttl) for complete details.

---

## Data Sources

- **Semantic Scholar API:** Paper metadata, citations, author info (200M+ papers)
- **arXiv:** Pre-print papers in CS.AI, CS.LG, CS.CL
- **Papers with Code:** Links to code repositories and datasets

---

## Project Structure

```
ml-research-kg/
├── schema/                           # RDF/OWL ontology definitions
├── src/
│   ├── extraction/                   # API clients (Semantic Scholar, arXiv, Papers with Code)
│   ├── kg/                           # RDF triple builder
│   ├── queries/                      # SPARQL queries
│   ├── ml/                           # Graph embeddings & link prediction
│   └── visualization/                # Network and embedding visualizations
├── scripts/
│   ├── main.py                       # Full pipeline orchestrator
│   └── viz_*.py                      # Standalone visualization scripts
├── backend/
│   └── server.py                     # FastAPI REST API
├── frontend/                         # React + Vite + D3 web interface
├── data/
│   ├── raw/                          # Raw JSON from API extraction
│   └── processed/                    # RDF triples (.ttl)
├── output/                           # Generated figures and HTML visualizations
└── docs/
```

---

## Installation & Usage

### Python pipeline

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline (extract → build KG → SPARQL → embed → visualize)
python scripts/main.py

# Skip API calls, rebuild from cached data/raw/ JSON
python scripts/main.py --skip-extraction

# Only run SPARQL queries on existing KG
python scripts/main.py --queries-only
```

Set `SEMANTIC_SCHOLAR_API_KEY` in your environment to avoid rate limiting (optional).

### Backend

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Milestones

- **Milestone 1 (Week 6):** ✅ Schema design complete
- **Milestone 2 (Week 13):** ✅ Working KG with 1000+ triples, SPARQL queries, graph embeddings
- **Milestone 3 (Week 15):** 📅 Final system with recommendation interface, interactive web interface

---

## Technologies

- **RDF/OWL + SPARQL:** Semantic data modeling and graph queries (RDFLib)
- **Python:** Data extraction, KG construction, ML pipeline
- **FastAPI:** Backend REST API serving the knowledge graph
- **React + Vite + D3:** Interactive frontend visualization
- **NetworkX / scikit-learn:** Graph analysis and SVD-based embeddings
- **PyKEEN (optional):** TransE/RotatE/ComplEx knowledge graph embeddings

---

## License

MIT License
