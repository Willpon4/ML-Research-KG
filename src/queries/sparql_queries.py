"""
SPARQL Queries for the ML Research Knowledge Graph
10 queries covering all 6 use cases.

Uses RDFLib's SPARQL implementation for local graph queries.
"""

from rdflib import Graph, Namespace

MLKG = Namespace("http://example.org/mlkg/")
MLKG_DATA = Namespace("http://example.org/mlkg/data/")


# ============================================================
# USE CASE 1: Paper Recommendation
# "Based on papers I've read, what should I read next?"
# ============================================================

QUERY_1A = """
# UC1a: Papers similar to a given paper (shared citations)
# Recommend papers that cite the same papers as a target paper.
# This finds papers with overlapping reference lists.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?recommended ?title (COUNT(?sharedRef) AS ?commonRefs) WHERE {
    # Target paper (parameterize this)
    ?target mlkg:title "Attention Is All You Need" .
    
    # Find what target cites
    ?target mlkg:cites ?sharedRef .
    
    # Find other papers that cite the same things
    ?recommended mlkg:cites ?sharedRef .
    ?recommended mlkg:title ?title .
    
    # Exclude the target itself
    FILTER(?recommended != ?target)
}
GROUP BY ?recommended ?title
ORDER BY DESC(?commonRefs)
LIMIT 10
"""

QUERY_1B = """
# UC1b: Papers recommended by shared topics
# Find papers on the same research topics as a given paper.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?paper ?title ?year ?citations (COUNT(?sharedTopic) AS ?topicOverlap) WHERE {
    # Target paper
    ?target mlkg:title "Attention Is All You Need" .
    ?target mlkg:hasKeyword ?sharedTopic .
    
    # Papers sharing topics
    ?paper mlkg:hasKeyword ?sharedTopic .
    ?paper mlkg:title ?title .
    ?paper mlkg:citationCount ?citations .
    
    OPTIONAL { ?paper mlkg:publicationYear ?year }
    
    FILTER(?paper != ?target)
}
GROUP BY ?paper ?title ?year ?citations
ORDER BY DESC(?topicOverlap) DESC(?citations)
LIMIT 10
"""


# ============================================================
# USE CASE 2: Emerging Trend Detection
# "What topics are getting attention right now?"
# ============================================================

QUERY_2 = """
# UC2: Emerging trends - topics with growing recent publication counts
# Identifies research topics with the most papers in recent years.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?topic ?topicName 
       (COUNT(?paper) AS ?paperCount)
       (AVG(?citations) AS ?avgCitations) WHERE {
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label ?topicName .
    ?paper mlkg:publicationYear ?year .
    ?paper mlkg:citationCount ?citations .
    
    # Recent papers only (2020+)
    FILTER(?year >= 2020)
}
GROUP BY ?topic ?topicName
ORDER BY DESC(?paperCount)
LIMIT 15
"""


# ============================================================
# USE CASE 3: Foundational Paper Discovery
# "What are the most important papers on transformers?"
# ============================================================

QUERY_3A = """
# UC3a: Most cited papers on a specific topic
# Find foundational/highly-cited papers about a research topic.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?paper ?title ?year ?citations ?venue WHERE {
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label "Transformers" .
    
    ?paper mlkg:title ?title .
    ?paper mlkg:citationCount ?citations .
    
    OPTIONAL { ?paper mlkg:publicationYear ?year }
    OPTIONAL { 
        ?paper mlkg:publishedIn ?v .
        ?v rdfs:label ?venue 
    }
}
ORDER BY DESC(?citations)
LIMIT 10
"""

QUERY_3B = """
# UC3b: Papers most cited BY other papers in the KG
# Finds papers that are referenced most within our knowledge graph.
# This measures influence within the collected corpus.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?paper ?title (COUNT(?citingPaper) AS ?inGraphCitations) WHERE {
    ?citingPaper mlkg:cites ?paper .
    ?paper mlkg:title ?title .
}
GROUP BY ?paper ?title
ORDER BY DESC(?inGraphCitations)
LIMIT 15
"""


# ============================================================
# USE CASE 4: Expert Discovery
# "Who are the leading researchers in graph neural networks?"
# ============================================================

QUERY_4 = """
# UC4: Top authors in a research area
# Rank authors by publication count and total citation impact
# in a specific research area.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?author ?authorName 
       (COUNT(DISTINCT ?paper) AS ?paperCount)
       (SUM(?citations) AS ?totalCitations)
       (GROUP_CONCAT(DISTINCT ?instName; separator=", ") AS ?affiliations) WHERE {
    ?author mlkg:authorOf ?paper .
    ?author rdfs:label ?authorName .
    
    ?paper mlkg:hasKeyword ?topic .
    ?topic rdfs:label ?topicLabel .
    FILTER(CONTAINS(LCASE(?topicLabel), "graph neural") || 
           CONTAINS(LCASE(?topicLabel), "knowledge graph"))
    
    ?paper mlkg:citationCount ?citations .
    
    OPTIONAL {
        ?author mlkg:affiliatedWith ?inst .
        ?inst rdfs:label ?instName .
    }
}
GROUP BY ?author ?authorName
ORDER BY DESC(?totalCitations)
LIMIT 10
"""


# ============================================================
# USE CASE 5: Research Timeline Tracking
# "How has deep learning evolved over time?"
# ============================================================

QUERY_5 = """
# UC5: Research evolution over time
# Track how the number of papers and citations in a research
# area change year over year.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?year 
       (COUNT(?paper) AS ?paperCount)
       (SUM(?citations) AS ?totalCitations)
       (AVG(?citations) AS ?avgCitations) WHERE {
    ?paper mlkg:inArea ?area .
    ?area rdfs:label ?areaName .
    FILTER(CONTAINS(LCASE(?areaName), "deep learning") || 
           CONTAINS(LCASE(?areaName), "machine learning"))
    
    ?paper mlkg:publicationYear ?year .
    ?paper mlkg:citationCount ?citations .
    
    FILTER(?year >= 2017 && ?year <= 2025)
}
GROUP BY ?year
ORDER BY ?year
"""


# ============================================================
# USE CASE 6: Collaboration Discovery
# "Who works together on similar topics?"
# ============================================================

QUERY_6A = """
# UC6a: Co-authorship network - most prolific collaborator pairs
# Find author pairs who have co-authored the most papers together.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?author1Name ?author2Name (COUNT(?paper) AS ?sharedPapers) WHERE {
    ?author1 mlkg:coauthorWith ?author2 .
    ?author1 rdfs:label ?author1Name .
    ?author2 rdfs:label ?author2Name .
    
    # Count shared papers
    ?author1 mlkg:authorOf ?paper .
    ?author2 mlkg:authorOf ?paper .
    
    # Avoid duplicate pairs (alphabetical ordering)
    FILTER(STR(?author1) < STR(?author2))
}
GROUP BY ?author1Name ?author2Name
ORDER BY DESC(?sharedPapers)
LIMIT 15
"""

QUERY_6B = """
# UC6b: Cross-institution collaborations
# Find papers authored by researchers from different institutions.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?inst1Name ?inst2Name (COUNT(DISTINCT ?paper) AS ?collabPapers) WHERE {
    ?a1 mlkg:authorOf ?paper .
    ?a2 mlkg:authorOf ?paper .
    
    ?a1 mlkg:affiliatedWith ?inst1 .
    ?a2 mlkg:affiliatedWith ?inst2 .
    
    ?inst1 rdfs:label ?inst1Name .
    ?inst2 rdfs:label ?inst2Name .
    
    FILTER(STR(?inst1) < STR(?inst2))
    FILTER(?a1 != ?a2)
}
GROUP BY ?inst1Name ?inst2Name
ORDER BY DESC(?collabPapers)
LIMIT 15
"""


# ============================================================
# BONUS: Overview / Summary query
# ============================================================

QUERY_OVERVIEW = """
# Overview: Knowledge graph statistics
# Count entities and relationships in the graph.

PREFIX mlkg: <http://example.org/mlkg/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT 
    (COUNT(DISTINCT ?pub) AS ?publications)
    (COUNT(DISTINCT ?author) AS ?authors)
    (COUNT(DISTINCT ?inst) AS ?institutions)
    (COUNT(DISTINCT ?venue) AS ?venues)
    (COUNT(DISTINCT ?area) AS ?researchAreas)
    (COUNT(DISTINCT ?topic) AS ?researchTopics)
WHERE {
    OPTIONAL { ?pub rdf:type mlkg:Publication }
    OPTIONAL { ?author rdf:type mlkg:Author }
    OPTIONAL { ?inst rdf:type mlkg:Institution }
    OPTIONAL { ?venue rdf:type mlkg:Venue }
    OPTIONAL { ?area rdf:type mlkg:ResearchArea }
    OPTIONAL { ?topic rdf:type mlkg:ResearchTopic }
}
"""


# ============================================================
# Query runner
# ============================================================

ALL_QUERIES = {
    "UC1a - Paper Recommendation (shared citations)": QUERY_1A,
    "UC1b - Paper Recommendation (shared topics)": QUERY_1B,
    "UC2  - Emerging Trend Detection": QUERY_2,
    "UC3a - Foundational Papers (by citation count)": QUERY_3A,
    "UC3b - Foundational Papers (in-graph influence)": QUERY_3B,
    "UC4  - Expert Discovery": QUERY_4,
    "UC5  - Research Timeline": QUERY_5,
    "UC6a - Collaboration (co-author pairs)": QUERY_6A,
    "UC6b - Collaboration (cross-institution)": QUERY_6B,
    "Overview - KG Statistics": QUERY_OVERVIEW,
}


def run_all_queries(graph, max_rows=10):
    """Run all queries against a graph and print results."""
    for name, query in ALL_QUERIES.items():
        print(f"\n{'=' * 70}")
        print(f"  {name}")
        print(f"{'=' * 70}")

        try:
            results = graph.query(query)
            rows = list(results)

            if not rows:
                print("  (No results)")
                continue

            # Print column headers
            if results.vars:
                headers = [str(v) for v in results.vars]
                print("  " + " | ".join(f"{h:>20s}" for h in headers))
                print("  " + "-" * (22 * len(headers)))

            # Print rows
            for i, row in enumerate(rows[:max_rows]):
                values = []
                for val in row:
                    s = str(val) if val else ""
                    # Truncate long strings
                    if len(s) > 50:
                        s = s[:47] + "..."
                    values.append(s)
                print("  " + " | ".join(f"{v:>20s}" for v in values))

            if len(rows) > max_rows:
                print(f"  ... ({len(rows) - max_rows} more rows)")

            print(f"  [{len(rows)} total results]")

        except Exception as e:
            print(f"  ERROR: {e}")


def run_single_query(graph, query_name):
    """Run a single named query."""
    if query_name in ALL_QUERIES:
        query = ALL_QUERIES[query_name]
        results = graph.query(query)
        return list(results)
    else:
        print(f"Unknown query: {query_name}")
        print(f"Available: {list(ALL_QUERIES.keys())}")
        return []


if __name__ == "__main__":
    # Load graph
    g = Graph()
    g.parse("data/processed/ml_research_kg.ttl", format="turtle")
    print(f"Loaded graph with {len(g)} triples")

    # Run all queries
    run_all_queries(g)
