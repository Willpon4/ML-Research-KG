"""
Multi-Source Data Merger
Combines data from Semantic Scholar, arXiv, and Papers with Code
into a unified paper/author collection.

Deduplicates papers across sources using arXiv IDs and title matching.
Enriches papers with data from multiple sources (e.g., code links from PwC).
"""

import json
import re
from pathlib import Path


def normalize_title(title):
    """Normalize a title for matching across sources."""
    if not title:
        return ""
    # Lowercase, strip punctuation, collapse whitespace
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_arxiv_id(paper):
    """Extract arXiv ID from a paper record in any source format."""
    # Direct arxivId field
    if "arxivId" in paper:
        return paper["arxivId"]

    # externalIds.ArXiv
    ext = paper.get("externalIds", {}) or {}
    if "ArXiv" in ext:
        return ext["ArXiv"]
    if "arxiv" in ext:
        return ext["arxiv"]

    # paperId starting with arxiv_
    pid = paper.get("paperId", "")
    if pid.startswith("arxiv_"):
        return pid[6:]

    return None


class MultiSourceMerger:
    """Merge papers from S2, arXiv, and Papers with Code."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.merged_papers = {}   # unified paperId -> paper
        self.merged_authors = {}  # authorId -> author
        self.source_counts = {"s2": 0, "arxiv": 0, "pwc": 0}
        self.merge_counts = {"merged": 0, "new": 0}

    def _load_json(self, filename):
        """Load JSON file if exists."""
        path = self.data_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def load_all_sources(self):
        """Load data from all three sources."""
        print("Loading data from all sources...")

        # Semantic Scholar
        self.s2_papers = self._load_json("s2_papers.json")
        self.s2_authors = self._load_json("s2_authors.json")
        print(f"  Semantic Scholar: {len(self.s2_papers)} papers, "
              f"{len(self.s2_authors)} authors")

        # arXiv
        self.arxiv_papers = self._load_json("arxiv_papers.json")
        self.arxiv_authors = self._load_json("arxiv_authors.json")
        print(f"  arXiv: {len(self.arxiv_papers)} papers, "
              f"{len(self.arxiv_authors)} authors")

        # Papers with Code
        self.pwc_papers = self._load_json("pwc_papers.json")
        self.pwc_code_links = self._load_json("pwc_code_links.json")
        print(f"  Papers with Code: {len(self.pwc_papers)} papers")

    def _build_lookup_maps(self):
        """
        Build lookup tables for deduplication.
        Maps: arxiv_id -> paperId, normalized_title -> paperId
        """
        arxiv_to_pid = {}
        title_to_pid = {}

        for pid, paper in self.merged_papers.items():
            arxiv_id = extract_arxiv_id(paper)
            if arxiv_id:
                arxiv_to_pid[arxiv_id] = pid

            norm_title = normalize_title(paper.get("title"))
            if norm_title and len(norm_title) > 15:  # Avoid short title collisions
                title_to_pid[norm_title] = pid

        return arxiv_to_pid, title_to_pid

    def _merge_paper_into_existing(self, existing_id, new_paper, source):
        """
        Merge data from new_paper into the existing paper record.
        Fills in missing fields rather than overwriting.
        """
        existing = self.merged_papers[existing_id]

        # Fill in missing scalar fields
        for field in ["abstract", "year", "venue", "url"]:
            if not existing.get(field) and new_paper.get(field):
                existing[field] = new_paper[field]

        # Citation count: take the maximum (S2 is usually most complete)
        existing_cc = existing.get("citationCount", 0) or 0
        new_cc = new_paper.get("citationCount", 0) or 0
        existing["citationCount"] = max(existing_cc, new_cc)

        # Merge author lists (by name, S2 data preferred)
        existing_names = {a.get("name", "").lower() for a in existing.get("authors", [])}
        for auth in new_paper.get("authors", []):
            if auth.get("name", "").lower() not in existing_names:
                existing.setdefault("authors", []).append(auth)
                existing_names.add(auth.get("name", "").lower())

        # Merge fields of study
        existing_fos = set(existing.get("fieldsOfStudy", []) or [])
        for f in (new_paper.get("fieldsOfStudy", []) or []):
            existing_fos.add(f)
        existing["fieldsOfStudy"] = list(existing_fos)

        # Merge external IDs
        existing.setdefault("externalIds", {}).update(new_paper.get("externalIds", {}) or {})

        # Code repositories (from PwC)
        if source == "pwc" and new_paper.get("code_repositories"):
            existing["code_repositories"] = new_paper["code_repositories"]

        # arXiv categories
        if source == "arxiv" and new_paper.get("arxivCategories"):
            existing["arxivCategories"] = new_paper["arxivCategories"]

        # PwC tasks/methods
        if source == "pwc":
            if new_paper.get("pwc_tasks"):
                existing["pwc_tasks"] = new_paper["pwc_tasks"]
            if new_paper.get("pwc_methods"):
                existing["pwc_methods"] = new_paper["pwc_methods"]

        # Track sources
        existing.setdefault("sources", []).append(source)

    def _add_new_paper(self, paper, source):
        """Add a new paper to the merged collection."""
        pid = paper.get("paperId")
        if not pid:
            return

        paper_copy = dict(paper)
        paper_copy["sources"] = [source]
        self.merged_papers[pid] = paper_copy

    def merge_source(self, source_papers, source_name):
        """
        Merge papers from one source into the unified collection.
        Uses arXiv ID and title matching for deduplication.
        """
        arxiv_to_pid, title_to_pid = self._build_lookup_maps()
        merged = 0
        new = 0

        for pid, paper in source_papers.items():
            # Try to find an existing match
            existing_id = None

            # 1. Match by arXiv ID
            arxiv_id = extract_arxiv_id(paper)
            if arxiv_id and arxiv_id in arxiv_to_pid:
                existing_id = arxiv_to_pid[arxiv_id]

            # 2. Match by normalized title
            if not existing_id:
                norm_title = normalize_title(paper.get("title"))
                if norm_title in title_to_pid:
                    existing_id = title_to_pid[norm_title]

            if existing_id:
                # Merge into existing
                self._merge_paper_into_existing(existing_id, paper, source_name)
                merged += 1
            else:
                # Add as new paper
                self._add_new_paper(paper, source_name)
                new += 1

                # Update lookup maps
                if arxiv_id:
                    arxiv_to_pid[arxiv_id] = pid
                norm_title = normalize_title(paper.get("title"))
                if norm_title and len(norm_title) > 15:
                    title_to_pid[norm_title] = pid

        self.merge_counts["merged"] += merged
        self.merge_counts["new"] += new

        print(f"  {source_name}: {new} new + {merged} merged")

    def merge_authors(self):
        """Merge authors from all sources, preferring richer data."""
        # Start with S2 authors (richest data: affiliations, hIndex)
        for aid, auth in self.s2_authors.items():
            self.merged_authors[aid] = auth

        # Add arXiv authors if not present (S2 has better data when overlap)
        for aid, auth in self.arxiv_authors.items():
            if aid not in self.merged_authors:
                # Check if this author exists under a different ID (by name)
                matched = False
                auth_name = auth.get("name", "").lower().strip()
                for existing in self.merged_authors.values():
                    if existing.get("name", "").lower().strip() == auth_name:
                        matched = True
                        break
                if not matched:
                    self.merged_authors[aid] = auth

        # PwC authors from paper records
        for paper in self.pwc_papers.values():
            for auth in paper.get("authors", []):
                aid = auth.get("authorId")
                if aid and aid not in self.merged_authors:
                    auth_name = auth.get("name", "").lower().strip()
                    matched = any(
                        existing.get("name", "").lower().strip() == auth_name
                        for existing in self.merged_authors.values()
                    )
                    if not matched:
                        self.merged_authors[aid] = auth

    def run_merge(self):
        """Run the full merge pipeline."""
        print("=" * 60)
        print("MERGING MULTI-SOURCE DATA")
        print("=" * 60)

        self.load_all_sources()

        print("\nMerging papers:")
        self.merge_source(self.s2_papers, "s2")
        self.merge_source(self.arxiv_papers, "arxiv")
        self.merge_source(self.pwc_papers, "pwc")

        print("\nMerging authors...")
        self.merge_authors()

        print(f"\n{'=' * 60}")
        print(f"MERGE COMPLETE")
        print(f"  Total unique papers: {len(self.merged_papers)}")
        print(f"  Total unique authors: {len(self.merged_authors)}")
        print(f"  Papers merged across sources: {self.merge_counts['merged']}")
        print(f"  New papers added: {self.merge_counts['new']}")

        # Source coverage
        source_stats = {"s2": 0, "arxiv": 0, "pwc": 0, "multi": 0}
        for paper in self.merged_papers.values():
            sources = set(paper.get("sources", []))
            if len(sources) > 1:
                source_stats["multi"] += 1
            for s in sources:
                if s in source_stats:
                    source_stats[s] += 1

        print(f"\n  Source coverage:")
        for src, count in source_stats.items():
            print(f"    {src}: {count}")
        print(f"{'=' * 60}")

    def save(self):
        """Save merged data."""
        papers_file = self.data_dir / "merged_papers.json"
        authors_file = self.data_dir / "merged_authors.json"

        with open(papers_file, "w") as f:
            json.dump(self.merged_papers, f, indent=2)
        print(f"\nSaved {len(self.merged_papers)} papers to {papers_file}")

        with open(authors_file, "w") as f:
            json.dump(self.merged_authors, f, indent=2)
        print(f"Saved {len(self.merged_authors)} authors to {authors_file}")


if __name__ == "__main__":
    merger = MultiSourceMerger(data_dir="data/raw")
    merger.run_merge()
    merger.save()
