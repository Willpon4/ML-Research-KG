# Machine Learning Research Knowledge Graph

A knowledge graph for exploring ML/AI research papers, enabling paper recommendations, trend detection, and research network analysis.

## Overview

This project builds a semantic knowledge graph that connects machine learning research papers, authors, institutions, citations, datasets, and code repositories. The system demonstrates knowledge graph engineering techniques including RDF/OWL schema design, SPARQL querying, and graph embeddings for recommendations.

**Course:** CS 497 - Knowledge Graph Engineering (15 weeks)  
**Technology Stack:** RDF/OWL, SPARQL, Python, AllegroGraph

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
├── README.md
├── schema/
│   ├── ml_research_ontology.ttl      # RDF/OWL ontology
│   └── sample_entities.md            # Example data
├── data/
│   ├── raw/                          # Raw API data
│   └── rdf/                          # RDF triples
├── src/
│   ├── extraction/                   # API clients
│   ├── queries/                      # SPARQL queries
│   └── ml/                          # Embeddings & link prediction
├── notebooks/                        # Jupyter demos
└── docs/
    └── milestones/                   # Project reports
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/willpon4/ml-research-kg.git
cd ml-research-kg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Milestones

- **Milestone 1 (Week 6):** ✅ Schema design complete
- **Milestone 2 (Week 13):** 🚧 Working KG with 1000+ triples, SPARQL queries, ML models
- **Milestone 3 (Week 15):** 📅 Final system with recommendation interface

---

## Technologies

- **RDF/OWL:** Semantic data modeling
- **SPARQL:** Graph query language
- **Python:** Data processing and ML
- **RDFLib:** RDF manipulation
- **PyKEEN:** Knowledge graph embeddings
- **AllegroGraph:** Triplestore

---

## Status

🚧 **In Development** - Milestone 1 Complete (Schema Design)

---

## License

MIT License

---

## Contact

**Project:** Knowledge Graph Engineering Course Project  
**GitHub:** [willpon4](https://github.com/willpon4)
