"""
arXiv Data Extractor
Uses the free arXiv API (no authentication, generous rate limits).

Pulls pre-print papers from CS.AI, CS.LG, CS.CL categories.
Returns paper metadata in a format compatible with the KG builder.
"""

import requests
import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

BASE_URL = "http://export.arxiv.org/api/query"
REQUEST_DELAY = 3.0  # arXiv asks for 3 seconds between requests

# XML namespaces used by arXiv
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# Mapping from arXiv categories to readable names
CATEGORY_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "stat.ML": "Statistics - Machine Learning",
}


class ArxivExtractor:
    """Extract paper data from the arXiv API."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers = {}   # arxivId -> paper data
        self.authors = {}  # name-based key -> author data
        self.request_count = 0

    def _request(self, params):
        """Make a rate-limited request to arXiv."""
        time.sleep(REQUEST_DELAY)
        self.request_count += 1

        url = f"{BASE_URL}?{urlencode(params)}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            else:
                print(f"  HTTP {resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            return None

    def _parse_entry(self, entry):
        """Parse a single arXiv entry element into a paper dict."""
        # arXiv ID (extract from URL)
        id_elem = entry.find("atom:id", NS)
        if id_elem is None:
            return None
        arxiv_url = id_elem.text
        arxiv_id = arxiv_url.split("/abs/")[-1]
        # Remove version suffix (v1, v2, etc.)
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        # Title
        title_elem = entry.find("atom:title", NS)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else None
        if title:
            title = re.sub(r"\s+", " ", title)

        # Abstract (summary)
        summary_elem = entry.find("atom:summary", NS)
        abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else None
        if abstract:
            abstract = re.sub(r"\s+", " ", abstract)

        # Published date
        published_elem = entry.find("atom:published", NS)
        year = None
        if published_elem is not None:
            try:
                year = int(published_elem.text[:4])
            except (ValueError, IndexError):
                pass

        # Authors
        authors = []
        for author_elem in entry.findall("atom:author", NS):
            name_elem = author_elem.find("atom:name", NS)
            if name_elem is not None and name_elem.text:
                name = name_elem.text.strip()
                # Use name-hash as author ID (arXiv doesn't provide stable IDs)
                author_id = f"arxiv_author_{re.sub(r'[^a-zA-Z0-9]', '_', name.lower())}"
                authors.append({
                    "authorId": author_id,
                    "name": name,
                    "affiliations": []  # arXiv rarely provides affiliations cleanly
                })

        # Categories (arXiv-specific)
        categories = []
        primary_cat = entry.find("arxiv:primary_category", NS)
        if primary_cat is not None:
            categories.append(primary_cat.get("term"))
        for cat_elem in entry.findall("atom:category", NS):
            term = cat_elem.get("term")
            if term and term not in categories:
                categories.append(term)

        # Readable fields of study
        fields_of_study = [
            CATEGORY_NAMES.get(cat, cat) for cat in categories
            if cat.startswith("cs.") or cat.startswith("stat.")
        ]

        # Venue: arXiv papers often don't have a published venue,
        # but we can note it as arXiv
        venue = "arXiv"

        # Construct paper in S2-compatible format
        paper = {
            "paperId": f"arxiv_{arxiv_id_clean}",
            "title": title,
            "abstract": abstract,
            "year": year,
            "citationCount": 0,  # arXiv doesn't track citations
            "authors": authors,
            "venue": venue,
            "fieldsOfStudy": fields_of_study,
            "externalIds": {"ArXiv": arxiv_id_clean},
            "url": f"https://arxiv.org/abs/{arxiv_id_clean}",
            "references": [],
            "citations": [],
            "arxivCategories": categories,
        }

        return paper

    def search_papers(self, query, max_results=100, start=0):
        """
        Search arXiv for papers matching a query.

        Args:
            query: Search query (arXiv query syntax)
            max_results: Maximum papers to fetch (paginates if >100)
            start: Starting offset for pagination
        """
        print(f"arXiv search: '{query}' (max={max_results})")

        papers_found = []
        batch_size = min(100, max_results)  # arXiv max per request
        current_start = start

        while len(papers_found) < max_results:
            remaining = max_results - len(papers_found)
            fetch = min(batch_size, remaining)

            params = {
                "search_query": query,
                "start": current_start,
                "max_results": fetch,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            xml_text = self._request(params)
            if not xml_text:
                break

            # Parse XML
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as e:
                print(f"  XML parse error: {e}")
                break

            entries = root.findall("atom:entry", NS)
            if not entries:
                break

            batch_papers = []
            for entry in entries:
                paper = self._parse_entry(entry)
                if paper and paper["paperId"] not in self.papers:
                    self.papers[paper["paperId"]] = paper
                    batch_papers.append(paper["paperId"])

                    # Extract authors
                    for auth in paper["authors"]:
                        aid = auth["authorId"]
                        if aid not in self.authors:
                            self.authors[aid] = auth

            papers_found.extend(batch_papers)
            print(f"  Got {len(batch_papers)} papers (total: {len(papers_found)})")

            if len(entries) < fetch:
                break  # No more results

            current_start += fetch

        return papers_found

    def build_collection(self, seed_queries, papers_per_query=60):
        """
        Build a collection of papers from seed queries.

        Args:
            seed_queries: list of arXiv search queries
            papers_per_query: how many to fetch per query
        """
        print("=" * 60)
        print("BUILDING arXiv COLLECTION")
        print("=" * 60)

        for query in seed_queries:
            self.search_papers(query, max_results=papers_per_query)
            print(f"  Cumulative papers: {len(self.papers)}")

        print(f"\n{'=' * 60}")
        print(f"arXiv COLLECTION COMPLETE")
        print(f"  Papers: {len(self.papers)}")
        print(f"  Authors: {len(self.authors)}")
        print(f"  API requests: {self.request_count}")
        print(f"{'=' * 60}")

    def save_data(self, filename_prefix="arxiv"):
        """Save collected data to JSON files."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        authors_file = self.data_dir / f"{filename_prefix}_authors.json"

        with open(papers_file, "w") as f:
            json.dump(self.papers, f, indent=2)
        print(f"Saved {len(self.papers)} papers to {papers_file}")

        with open(authors_file, "w") as f:
            json.dump(self.authors, f, indent=2)
        print(f"Saved {len(self.authors)} authors to {authors_file}")

    def load_data(self, filename_prefix="arxiv"):
        """Load previously saved data."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        authors_file = self.data_dir / f"{filename_prefix}_authors.json"

        if papers_file.exists():
            with open(papers_file) as f:
                self.papers = json.load(f)
            print(f"Loaded {len(self.papers)} papers")

        if authors_file.exists():
            with open(authors_file) as f:
                self.authors = json.load(f)
            print(f"Loaded {len(self.authors)} authors")


# ============================================================
# Default seed queries for ML research
# Uses arXiv's query syntax
# ============================================================
DEFAULT_ARXIV_QUERIES = [
    # Transformers and LLMs
    'cat:cs.CL AND (abs:transformer OR abs:"attention mechanism")',
    'cat:cs.CL AND (abs:"large language model" OR abs:GPT OR abs:BERT)',
    # Computer Vision
    'cat:cs.CV AND (abs:"object detection" OR abs:"image classification")',
    'cat:cs.CV AND (abs:"generative adversarial" OR abs:GAN)',
    'cat:cs.CV AND abs:"diffusion model"',
    # Graph ML
    'cat:cs.LG AND (abs:"graph neural network" OR abs:GNN)',
    'cat:cs.LG AND abs:"knowledge graph"',
    # Reinforcement Learning
    'cat:cs.LG AND (abs:"reinforcement learning" OR abs:"policy gradient")',
    # Self-supervised / Representation Learning
    'cat:cs.LG AND (abs:"self-supervised" OR abs:"contrastive learning")',
    # Federated Learning
    'cat:cs.LG AND abs:"federated learning"',
    # Neural Architecture Search
    'cat:cs.LG AND abs:"neural architecture search"',
    # Foundation models
    'cat:cs.LG AND (abs:"foundation model" OR abs:"pre-trained")',
]


if __name__ == "__main__":
    extractor = ArxivExtractor(data_dir="data/raw")

    extractor.build_collection(
        seed_queries=DEFAULT_ARXIV_QUERIES,
        papers_per_query=60
    )

    extractor.save_data()
