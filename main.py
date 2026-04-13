"""
ML Research Knowledge Graph - Main Pipeline
============================================
Runs the complete workflow:
1. Extract data from Semantic Scholar API
2. Build RDF Knowledge Graph
3. Run SPARQL queries (all 6 use cases)
4. Train graph embeddings
5. Evaluate link prediction
6. Generate visualizations

Usage:
    python main.py                    # Run full pipeline
    python main.py --skip-extraction  # Skip API calls, use existing data
    python main.py --queries-only     # Just run SPARQL queries on existing KG
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_extraction(data_dir="data/raw", skip_s2=False):
    """Phase 1: Extract data from all three sources."""
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA EXTRACTION (3 sources)")
    print("=" * 70)

    # 1a: arXiv (fast, no rate limit issues)
    print("\n--- 1a: arXiv ---")
    arxiv_papers_file = Path(data_dir) / "arxiv_papers.json"
    if arxiv_papers_file.exists():
        print(f"Found existing arXiv data, skipping")
    else:
        from extraction.arxiv_extractor import ArxivExtractor, DEFAULT_ARXIV_QUERIES
        arxiv = ArxivExtractor(data_dir=data_dir)
        arxiv.build_collection(
            seed_queries=DEFAULT_ARXIV_QUERIES,
            papers_per_query=60
        )
        arxiv.save_data()

    # 1b: Papers with Code (downloads static files)
    print("\n--- 1b: Papers with Code ---")
    pwc_papers_file = Path(data_dir) / "pwc_papers.json"
    if pwc_papers_file.exists():
        print(f"Found existing PwC data, skipping")
    else:
        from extraction.papers_with_code import PapersWithCodeExtractor
        pwc = PapersWithCodeExtractor(data_dir=data_dir)
        try:
            pwc.extract_papers_with_code(limit=500)
            pwc.save_data()
        except Exception as e:
            print(f"  PwC extraction failed: {e}")
            print("  Continuing without PwC data...")

    # 1c: Semantic Scholar (slow due to rate limits)
    if not skip_s2:
        print("\n--- 1c: Semantic Scholar ---")
        s2_papers_file = Path(data_dir) / "s2_papers.json"
        if s2_papers_file.exists():
            print(f"Found existing S2 data, skipping")
        else:
            from extraction.semantic_scholar import SemanticScholarExtractor, DEFAULT_SEED_QUERIES
            s2 = SemanticScholarExtractor(data_dir=data_dir)
            s2.build_seed_collection(
                seed_queries=DEFAULT_SEED_QUERIES,
                papers_per_query=50,
                follow_citations=True,
                citation_depth=1,
                citations_per_paper=10
            )
            s2.extract_authors_from_papers()
            s2.save_data()
    else:
        print("\n--- 1c: Semantic Scholar SKIPPED ---")

    # 1d: Merge all sources
    print("\n--- 1d: Merging all sources ---")
    from extraction.merger import MultiSourceMerger
    merger = MultiSourceMerger(data_dir=data_dir)
    merger.run_merge()
    merger.save()

    return merger


def run_kg_build(data_dir="data/raw", output_dir="data/processed",
                  schema_path="schema/ontology.ttl"):
    """Phase 2: Build the RDF Knowledge Graph from merged data."""
    from kg.builder import KnowledgeGraphBuilder

    print("\n" + "=" * 70)
    print("  PHASE 2: KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 70)

    builder = KnowledgeGraphBuilder()

    # Load ontology
    if Path(schema_path).exists():
        builder.load_ontology(schema_path)

    # Prefer merged data if available, else fall back to S2
    merged_papers = Path(data_dir) / "merged_papers.json"
    merged_authors = Path(data_dir) / "merged_authors.json"
    s2_papers = Path(data_dir) / "s2_papers.json"
    s2_authors = Path(data_dir) / "s2_authors.json"

    if merged_papers.exists():
        print("Using merged multi-source data")
        papers_file = str(merged_papers)
        authors_file = str(merged_authors)
    else:
        print("Merged data not found, using S2 data only")
        papers_file = str(s2_papers)
        authors_file = str(s2_authors)

    builder.build_from_extracted_data(
        papers_file=papers_file,
        authors_file=authors_file
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "ml_research_kg.ttl"
    builder.save_graph(str(output_path))

    # Print stats
    builder.get_stats()

    return builder


def run_queries(kg_path="data/processed/ml_research_kg.ttl"):
    """Phase 3: Run all SPARQL queries."""
    from rdflib import Graph
    from queries.sparql_queries import run_all_queries

    print("\n" + "=" * 70)
    print("  PHASE 3: SPARQL QUERIES (6 USE CASES)")
    print("=" * 70)

    g = Graph()
    g.parse(kg_path, format="turtle")
    print(f"Loaded graph with {len(g)} triples")

    run_all_queries(g, max_rows=10)

    return g


def run_ml_pipeline(kg_path="data/processed/ml_research_kg.ttl",
                     output_dir="output"):
    """Phase 4: Graph embeddings and link prediction."""
    from ml.embeddings import run_full_pipeline

    print("\n" + "=" * 70)
    print("  PHASE 4: GRAPH EMBEDDINGS & LINK PREDICTION")
    print("=" * 70)

    pipe = run_full_pipeline(
        kg_path=kg_path,
        output_dir=output_dir,
        model_name="TransE",
        embedding_dim=128,
        num_epochs=100
    )

    return pipe


def run_visualizations(kg_path="data/processed/ml_research_kg.ttl",
                        output_dir="output"):
    """Phase 5: Generate all visualizations."""
    from visualization.visualizations import KGVisualizer

    print("\n" + "=" * 70)
    print("  PHASE 5: VISUALIZATIONS")
    print("=" * 70)

    viz = KGVisualizer(kg_path, output_dir)
    viz.generate_all()
    return viz


def main():
    parser = argparse.ArgumentParser(
        description="ML Research Knowledge Graph Pipeline"
    )
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip all extraction, use existing data")
    parser.add_argument("--skip-s2", action="store_true",
                        help="Skip Semantic Scholar (use arXiv + PwC only — fast, no API key needed)")
    parser.add_argument("--queries-only", action="store_true",
                        help="Only run SPARQL queries on existing KG")
    parser.add_argument("--ml-only", action="store_true",
                        help="Only run ML pipeline on existing KG")
    parser.add_argument("--viz-only", action="store_true",
                        help="Only generate visualizations from existing KG")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory for raw data")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for outputs")

    args = parser.parse_args()

    print("=" * 70)
    print("  ML RESEARCH KNOWLEDGE GRAPH")
    print("  Mapping the Machine Learning Research Landscape")
    print("=" * 70)

    kg_path = "data/processed/ml_research_kg.ttl"

    if args.queries_only:
        run_queries(kg_path)
        return

    if args.ml_only:
        run_ml_pipeline(kg_path, args.output_dir)
        return

    if args.viz_only:
        run_visualizations(kg_path, args.output_dir)
        return

    # Full pipeline
    if not args.skip_extraction:
        run_extraction(args.data_dir, skip_s2=args.skip_s2)

    run_kg_build(args.data_dir, "data/processed")
    run_queries(kg_path)
    run_ml_pipeline(kg_path, args.output_dir)
    run_visualizations(kg_path, args.output_dir)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Knowledge Graph: {kg_path}")
    print(f"  Visualizations:  {args.output_dir}/")
    print(f"  Metrics:         {args.output_dir}/link_prediction_results.json")


if __name__ == "__main__":
    main()
