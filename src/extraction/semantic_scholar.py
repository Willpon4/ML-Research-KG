"""
Semantic Scholar API Data Extractor
Primary data source for the ML Research Knowledge Graph.

Pulls paper metadata, citations, author info, and abstracts
from the Semantic Scholar API.

API Key: set environment variable S2_API_KEY for higher rate limits.
Without a key: ~1 request per 5-10 seconds (shared pool)
With a key: 1 request per second (dedicated)
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Rate limiting: adjusts based on whether API key is set
API_KEY = os.environ.get("S2_API_KEY")
REQUEST_DELAY = 1.1 if API_KEY else 3.0  # seconds between requests

# Fields to request from the API
PAPER_FIELDS = [
    "paperId", "title", "abstract", "year", "citationCount",
    "authors", "venue", "fieldsOfStudy", "citations",
    "references", "externalIds", "url"
]

AUTHOR_FIELDS = [
    "authorId", "name", "affiliations", "paperCount",
    "citationCount", "hIndex"
]


class SemanticScholarExtractor:
    """Extract paper and author data from Semantic Scholar API."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers = {}       # paperId -> paper data
        self.authors = {}      # authorId -> author data
        self.request_count = 0

    def _request(self, url, params=None):
        """Make a rate-limited request to the S2 API."""
        time.sleep(REQUEST_DELAY)
        self.request_count += 1

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print(f"  Rate limited. Waiting 60s...")
                time.sleep(60)
                return self._request(url, params)
            else:
                print(f"  HTTP {resp.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            return None

    def search_papers(self, query, limit=100):
        """
        Search for papers by keyword query.
        Returns list of paper IDs.
        """
        print(f"Searching for: '{query}' (limit={limit})")
        papers_found = []
        offset = 0
        batch_size = min(limit, 100)  # API max per request

        while offset < limit:
            url = f"{BASE_URL}/paper/search"
            params = {
                "query": query,
                "offset": offset,
                "limit": batch_size,
                "fields": ",".join(PAPER_FIELDS)
            }

            data = self._request(url, params)
            if not data or "data" not in data:
                break

            batch = data["data"]
            if not batch:
                break

            for paper in batch:
                pid = paper.get("paperId")
                if pid and pid not in self.papers:
                    self.papers[pid] = paper
                    papers_found.append(pid)

            print(f"  Fetched {len(batch)} papers (total: {len(papers_found)})")
            offset += batch_size

            if len(batch) < batch_size:
                break

        return papers_found

    def get_paper_details(self, paper_id):
        """Fetch full details for a single paper."""
        if paper_id in self.papers and self.papers[paper_id].get("abstract"):
            return self.papers[paper_id]

        url = f"{BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(PAPER_FIELDS)}

        data = self._request(url, params)
        if data:
            self.papers[paper_id] = data
        return data

    def get_paper_citations(self, paper_id, limit=50):
        """Get papers that cite a given paper."""
        url = f"{BASE_URL}/paper/{paper_id}/citations"
        params = {
            "fields": "paperId,title,year,citationCount,authors,venue",
            "limit": limit
        }

        data = self._request(url, params)
        if data and "data" in data:
            citing_papers = []
            for item in data["data"]:
                citing = item.get("citingPaper", {})
                pid = citing.get("paperId")
                if pid:
                    if pid not in self.papers:
                        self.papers[pid] = citing
                    citing_papers.append(pid)
            return citing_papers
        return []

    def get_paper_references(self, paper_id, limit=50):
        """Get papers referenced by a given paper."""
        url = f"{BASE_URL}/paper/{paper_id}/references"
        params = {
            "fields": "paperId,title,year,citationCount,authors,venue",
            "limit": limit
        }

        data = self._request(url, params)
        if data and "data" in data:
            ref_papers = []
            for item in data["data"]:
                cited = item.get("citedPaper", {})
                pid = cited.get("paperId")
                if pid:
                    if pid not in self.papers:
                        self.papers[pid] = cited
                    ref_papers.append(pid)
            return ref_papers
        return []

    def get_author_details(self, author_id):
        """Fetch details for an author."""
        if author_id in self.authors:
            return self.authors[author_id]

        url = f"{BASE_URL}/author/{author_id}"
        params = {"fields": ",".join(AUTHOR_FIELDS)}

        data = self._request(url, params)
        if data:
            self.authors[author_id] = data
        return data

    def extract_authors_from_papers(self):
        """Pull author details for all authors found in collected papers."""
        author_ids = set()
        for paper in self.papers.values():
            for author in paper.get("authors", []):
                aid = author.get("authorId")
                if aid:
                    author_ids.add(aid)

        print(f"\nExtracting details for {len(author_ids)} unique authors...")
        count = 0
        for aid in author_ids:
            if aid not in self.authors:
                self.get_author_details(aid)
                count += 1
                if count % 10 == 0:
                    print(f"  Processed {count}/{len(author_ids)} authors")

        return list(author_ids)

    def build_seed_collection(self, seed_queries, papers_per_query=50,
                               follow_citations=True, citation_depth=1,
                               citations_per_paper=10):
        """
        Build a collection of papers starting from seed search queries.
        Optionally follows citation links to expand coverage.

        Args:
            seed_queries: list of search query strings
            papers_per_query: how many papers to fetch per query
            follow_citations: whether to follow citation/reference links
            citation_depth: how many hops to follow
            citations_per_paper: max citations/references to follow per paper
        """
        print("=" * 60)
        print("BUILDING SEED COLLECTION")
        print("=" * 60)

        # Phase 1: Search queries
        seed_paper_ids = []
        for query in seed_queries:
            ids = self.search_papers(query, limit=papers_per_query)
            seed_paper_ids.extend(ids)
            print(f"  Total papers so far: {len(self.papers)}")

        # Phase 2: Follow citations/references to expand
        if follow_citations:
            current_ids = list(seed_paper_ids)
            for depth in range(citation_depth):
                print(f"\n--- Citation expansion depth {depth + 1} ---")
                new_ids = []
                for i, pid in enumerate(current_ids[:30]):  # Limit expansion
                    refs = self.get_paper_references(pid, limit=citations_per_paper)
                    new_ids.extend([r for r in refs if r not in self.papers])

                    if i % 10 == 0:
                        print(f"  Expanded {i}/{min(len(current_ids), 30)} papers, "
                              f"total: {len(self.papers)}")

                current_ids = new_ids

        # Phase 3: Get full details for papers missing abstracts
        print(f"\n--- Fetching full details for papers missing data ---")
        papers_needing_details = [
            pid for pid, p in self.papers.items()
            if not p.get("abstract") and p.get("title")
        ]
        for i, pid in enumerate(papers_needing_details[:100]):  # Cap at 100
            self.get_paper_details(pid)
            if i % 10 == 0:
                print(f"  Detailed {i}/{min(len(papers_needing_details), 100)}")

        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE")
        print(f"  Papers: {len(self.papers)}")
        print(f"  Authors: {len(self.authors)}")
        print(f"  API requests made: {self.request_count}")
        print(f"{'=' * 60}")

    def save_data(self, filename_prefix="s2"):
        """Save collected data to JSON files."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        authors_file = self.data_dir / f"{filename_prefix}_authors.json"

        with open(papers_file, "w") as f:
            json.dump(self.papers, f, indent=2)
        print(f"Saved {len(self.papers)} papers to {papers_file}")

        with open(authors_file, "w") as f:
            json.dump(self.authors, f, indent=2)
        print(f"Saved {len(self.authors)} authors to {authors_file}")

    def load_data(self, filename_prefix="s2"):
        """Load previously saved data."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        authors_file = self.data_dir / f"{filename_prefix}_authors.json"

        if papers_file.exists():
            with open(papers_file) as f:
                self.papers = json.load(f)
            print(f"Loaded {len(self.papers)} papers from {papers_file}")

        if authors_file.exists():
            with open(authors_file) as f:
                self.authors = json.load(f)
            print(f"Loaded {len(self.authors)} authors from {authors_file}")

    def get_stats(self):
        """Print collection statistics."""
        total_papers = len(self.papers)
        papers_with_abstract = sum(
            1 for p in self.papers.values() if p.get("abstract")
        )
        papers_with_venue = sum(
            1 for p in self.papers.values() if p.get("venue")
        )
        papers_with_year = sum(
            1 for p in self.papers.values() if p.get("year")
        )
        total_authors = len(self.authors)
        total_author_refs = sum(
            len(p.get("authors", [])) for p in self.papers.values()
        )

        print(f"\n--- Collection Statistics ---")
        print(f"Papers: {total_papers}")
        print(f"  With abstract: {papers_with_abstract} "
              f"({papers_with_abstract/max(total_papers,1)*100:.0f}%)")
        print(f"  With venue: {papers_with_venue} "
              f"({papers_with_venue/max(total_papers,1)*100:.0f}%)")
        print(f"  With year: {papers_with_year} "
              f"({papers_with_year/max(total_papers,1)*100:.0f}%)")
        print(f"Authors (unique IDs): {total_authors}")
        print(f"Author-paper links: {total_author_refs}")


# ============================================================
# Default seed queries covering major ML research areas
# ============================================================
DEFAULT_SEED_QUERIES = [
    "transformer architecture attention mechanism",
    "large language models GPT",
    "graph neural networks",
    "reinforcement learning deep",
    "generative adversarial networks",
    "diffusion models image generation",
    "computer vision object detection",
    "natural language processing BERT",
    "knowledge graph embedding",
    "federated learning privacy",
    "self-supervised learning representation",
    "neural architecture search",
]


if __name__ == "__main__":
    extractor = SemanticScholarExtractor(data_dir="data/raw")

    # Build collection from seed queries
    extractor.build_seed_collection(
        seed_queries=DEFAULT_SEED_QUERIES,
        papers_per_query=50,
        follow_citations=True,
        citation_depth=1,
        citations_per_paper=10
    )

    # Extract author details
    extractor.extract_authors_from_papers()

    # Save raw data
    extractor.save_data()

    # Print stats
    extractor.get_stats()
