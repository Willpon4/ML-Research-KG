"""
Papers with Code Extractor
Uses Papers with Code's public data dumps (JSON files on their website).

Adds code repository links, datasets, and methods to ML papers.
No API rate limits since we download static files.
"""

import requests
import json
import gzip
import io
from pathlib import Path

# Papers with Code publishes JSON dumps at these URLs
PWC_URLS = {
    "papers_with_abstracts": "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz",
    "links_between_papers_and_code": "https://production-media.paperswithcode.com/about/links-between-papers-and-code.json.gz",
    "methods": "https://production-media.paperswithcode.com/about/methods.json.gz",
    "datasets": "https://production-media.paperswithcode.com/about/datasets.json.gz",
}


class PapersWithCodeExtractor:
    """Extract paper-code-dataset links from Papers with Code."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers = {}           # paperId -> paper data
        self.code_links = {}       # paperId -> list of repo URLs
        self.paper_datasets = {}   # paperId -> list of dataset names

    def download_dataset(self, name, url):
        """Download and parse a PwC data dump."""
        local_path = self.data_dir / f"pwc_{name}.json"

        # Check cache
        if local_path.exists():
            print(f"  Using cached {local_path}")
            with open(local_path) as f:
                return json.load(f)

        print(f"  Downloading {name}...")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} for {url}")
                return None

            # Decompress gzip
            compressed = io.BytesIO(resp.content)
            with gzip.open(compressed, "rt", encoding="utf-8") as f:
                data = json.load(f)

            # Cache uncompressed
            with open(local_path, "w") as f:
                json.dump(data, f)

            print(f"  Downloaded {name}: {len(data) if isinstance(data, list) else 'OK'}")
            return data
        except Exception as e:
            print(f"  Error downloading {name}: {e}")
            return None

    def extract_papers_with_code(self, limit=500):
        """
        Extract papers that have associated code.

        Args:
            limit: maximum papers to keep
        """
        print("=" * 60)
        print("EXTRACTING FROM PAPERS WITH CODE")
        print("=" * 60)

        # Download paper-code links
        links_data = self.download_dataset(
            "links", PWC_URLS["links_between_papers_and_code"]
        )
        if not links_data:
            print("Could not download code links")
            return

        print(f"\nProcessing {len(links_data)} paper-code links...")

        # Build code links dict: paper_url -> list of repo URLs
        paper_repos = {}
        for link in links_data:
            paper_url = link.get("paper_url") or link.get("paper_arxiv_id") or link.get("paper_title")
            repo_url = link.get("repo_url")
            if paper_url and repo_url:
                paper_repos.setdefault(paper_url, []).append(repo_url)

        print(f"  Found {len(paper_repos)} papers with code")

        # Download paper metadata
        papers_data = self.download_dataset(
            "papers", PWC_URLS["papers_with_abstracts"]
        )
        if not papers_data:
            print("Could not download paper metadata")
            return

        print(f"\nProcessing {len(papers_data)} papers...")

        # Filter to papers that have code and build our dataset
        count = 0
        for paper in papers_data:
            if count >= limit:
                break

            paper_url = paper.get("paper_url")
            arxiv_id = paper.get("arxiv_id")
            title = paper.get("title")

            if not title:
                continue

            # Check if this paper has associated code
            repos = []
            if paper_url in paper_repos:
                repos = paper_repos[paper_url]
            if arxiv_id and f"https://arxiv.org/abs/{arxiv_id}" in paper_repos:
                repos.extend(paper_repos[f"https://arxiv.org/abs/{arxiv_id}"])

            if not repos:
                continue

            # Use arxiv_id as paper ID if available, else hash of title
            if arxiv_id:
                paper_id = f"arxiv_{arxiv_id}"
            else:
                paper_id = f"pwc_{abs(hash(title))}"

            # Extract tasks/methods as research topics
            tasks = paper.get("tasks", []) or []
            methods = paper.get("methods", []) or []

            # Convert to our format
            paper_entry = {
                "paperId": paper_id,
                "title": title,
                "abstract": paper.get("abstract") or "",
                "year": self._extract_year(paper.get("date")),
                "citationCount": 0,
                "authors": [
                    {"authorId": f"pwc_author_{abs(hash(a))}", "name": a, "affiliations": []}
                    for a in (paper.get("authors", []) or [])
                    if a
                ],
                "venue": paper.get("proceeding") or "arXiv",
                "fieldsOfStudy": tasks[:5],
                "externalIds": {
                    "ArXiv": arxiv_id,
                    "DOI": paper.get("doi"),
                } if arxiv_id else {},
                "url": paper_url,
                "references": [],
                "citations": [],
                "pwc_tasks": tasks,
                "pwc_methods": methods,
                "code_repositories": repos,
            }

            self.papers[paper_id] = paper_entry
            self.code_links[paper_id] = repos
            count += 1

        print(f"\n  Extracted {len(self.papers)} papers with code links")

        # Summary
        total_repos = sum(len(repos) for repos in self.code_links.values())
        print(f"  Total code repositories: {total_repos}")

    def _extract_year(self, date_str):
        """Extract year from a date string like '2022-01-15'."""
        if not date_str:
            return None
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    def save_data(self, filename_prefix="pwc"):
        """Save collected data to JSON files."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        with open(papers_file, "w") as f:
            json.dump(self.papers, f, indent=2)
        print(f"Saved {len(self.papers)} papers to {papers_file}")

        links_file = self.data_dir / f"{filename_prefix}_code_links.json"
        with open(links_file, "w") as f:
            json.dump(self.code_links, f, indent=2)
        print(f"Saved code links to {links_file}")

    def load_data(self, filename_prefix="pwc"):
        """Load previously saved data."""
        papers_file = self.data_dir / f"{filename_prefix}_papers.json"
        links_file = self.data_dir / f"{filename_prefix}_code_links.json"

        if papers_file.exists():
            with open(papers_file) as f:
                self.papers = json.load(f)
            print(f"Loaded {len(self.papers)} PwC papers")

        if links_file.exists():
            with open(links_file) as f:
                self.code_links = json.load(f)
            print(f"Loaded {len(self.code_links)} code links")


if __name__ == "__main__":
    extractor = PapersWithCodeExtractor(data_dir="data/raw")
    extractor.extract_papers_with_code(limit=500)
    extractor.save_data()
