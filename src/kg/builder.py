"""
Knowledge Graph Builder
Converts extracted paper/author data into RDF triples.

Generates triples following the mlkg ontology schema:
8 classes, 12 object properties, 4 datatype properties.
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD


# Namespaces
MLKG = Namespace("http://example.org/mlkg/")
MLKG_DATA = Namespace("http://example.org/mlkg/data/")


# ============================================================
# Topic and area mappings for keyword extraction
# ============================================================

RESEARCH_AREAS = {
    "natural_language_processing": {
        "label": "Natural Language Processing",
        "keywords": ["nlp", "natural language", "text", "language model",
                      "translation", "sentiment", "named entity", "parsing",
                      "question answering", "summarization", "dialogue"]
    },
    "computer_vision": {
        "label": "Computer Vision",
        "keywords": ["image", "visual", "object detection", "segmentation",
                      "video", "face", "pose", "optical", "scene", "3d",
                      "image generation", "image synthesis"]
    },
    "deep_learning": {
        "label": "Deep Learning",
        "keywords": ["deep learning", "neural network", "deep neural",
                      "backpropagation", "activation function", "layer"]
    },
    "reinforcement_learning": {
        "label": "Reinforcement Learning",
        "keywords": ["reinforcement learning", "reward", "policy gradient",
                      "q-learning", "multi-agent", "exploration"]
    },
    "graph_ml": {
        "label": "Graph Machine Learning",
        "keywords": ["graph neural", "graph network", "knowledge graph",
                      "graph embedding", "node classification", "link prediction"]
    },
    "generative_models": {
        "label": "Generative Models",
        "keywords": ["generative", "gan", "vae", "diffusion", "autoregressive",
                      "image generation", "text generation", "synthesis"]
    },
    "optimization": {
        "label": "Optimization and Training",
        "keywords": ["optimization", "gradient descent", "learning rate",
                      "batch", "regularization", "dropout", "convergence"]
    },
    "representation_learning": {
        "label": "Representation Learning",
        "keywords": ["representation learning", "embedding", "self-supervised",
                      "contrastive", "pretrain", "pre-train", "transfer learning",
                      "foundation model"]
    }
}

RESEARCH_TOPICS = {
    "transformer": {
        "label": "Transformers",
        "keywords": ["transformer", "attention mechanism", "self-attention",
                      "multi-head attention"],
        "area": "deep_learning"
    },
    "bert": {
        "label": "BERT and Variants",
        "keywords": ["bert", "roberta", "electra", "deberta", "masked language"],
        "area": "natural_language_processing"
    },
    "gpt": {
        "label": "GPT and Large Language Models",
        "keywords": ["gpt", "large language model", "llm", "chatgpt",
                      "instruction tuning", "rlhf", "in-context learning"],
        "area": "natural_language_processing"
    },
    "diffusion": {
        "label": "Diffusion Models",
        "keywords": ["diffusion model", "denoising", "score matching",
                      "stable diffusion", "ddpm", "dalle"],
        "area": "generative_models"
    },
    "gan": {
        "label": "Generative Adversarial Networks",
        "keywords": ["gan", "generative adversarial", "discriminator",
                      "generator", "stylegan", "cyclegan"],
        "area": "generative_models"
    },
    "gnn": {
        "label": "Graph Neural Networks",
        "keywords": ["graph neural network", "gnn", "graph convolutional",
                      "message passing", "node embedding", "graph attention"],
        "area": "graph_ml"
    },
    "cnn": {
        "label": "Convolutional Neural Networks",
        "keywords": ["convolutional neural", "cnn", "resnet", "convolution",
                      "pooling", "feature map"],
        "area": "computer_vision"
    },
    "object_detection": {
        "label": "Object Detection",
        "keywords": ["object detection", "yolo", "faster rcnn", "anchor",
                      "bounding box", "detection"],
        "area": "computer_vision"
    },
    "knowledge_graph": {
        "label": "Knowledge Graphs",
        "keywords": ["knowledge graph", "knowledge base", "triple",
                      "entity embedding", "relation prediction", "kg embedding"],
        "area": "graph_ml"
    },
    "federated_learning": {
        "label": "Federated Learning",
        "keywords": ["federated learning", "federated", "differential privacy",
                      "distributed learning", "privacy-preserving"],
        "area": "optimization"
    },
    "rl_deep": {
        "label": "Deep Reinforcement Learning",
        "keywords": ["deep reinforcement", "dqn", "ppo", "a3c", "actor-critic",
                      "policy optimization"],
        "area": "reinforcement_learning"
    },
    "self_supervised": {
        "label": "Self-Supervised Learning",
        "keywords": ["self-supervised", "contrastive learning", "simclr",
                      "moco", "byol", "pretext task"],
        "area": "representation_learning"
    },
    "neural_arch_search": {
        "label": "Neural Architecture Search",
        "keywords": ["neural architecture search", "nas", "automl",
                      "architecture search"],
        "area": "optimization"
    },
    "transfer_learning": {
        "label": "Transfer Learning",
        "keywords": ["transfer learning", "domain adaptation", "fine-tuning",
                      "pretrained", "pre-trained"],
        "area": "representation_learning"
    },
    "attention": {
        "label": "Attention Mechanisms",
        "keywords": ["attention", "cross-attention", "self-attention",
                      "multi-head", "flash attention"],
        "area": "deep_learning"
    }
}


def make_uri(entity_type, identifier):
    """Create a clean URI from an entity type and identifier."""
    clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(identifier))
    return MLKG_DATA[f"{entity_type}/{clean_id}"]


def make_hash_uri(entity_type, text):
    """Create a URI using a hash of the text (for topics, areas, etc.)."""
    clean = re.sub(r'[^a-zA-Z0-9]', '_', text.lower().strip())
    return MLKG_DATA[f"{entity_type}/{clean}"]


def match_topics(text):
    """Match research topics from text (title + abstract)."""
    if not text:
        return []
    text_lower = text.lower()
    matched = []
    for topic_id, topic_info in RESEARCH_TOPICS.items():
        for kw in topic_info["keywords"]:
            if kw.lower() in text_lower:
                matched.append(topic_id)
                break
    return matched


def match_areas(text, matched_topics=None):
    """Match research areas from text and/or matched topics."""
    areas = set()

    # Areas from matched topics
    if matched_topics:
        for tid in matched_topics:
            if tid in RESEARCH_TOPICS:
                areas.add(RESEARCH_TOPICS[tid]["area"])

    # Direct area matching from text
    if text:
        text_lower = text.lower()
        for area_id, area_info in RESEARCH_AREAS.items():
            for kw in area_info["keywords"]:
                if kw.lower() in text_lower:
                    areas.add(area_id)
                    break

    return list(areas)


class KnowledgeGraphBuilder:
    """Build RDF knowledge graph from extracted paper data."""

    def __init__(self):
        self.g = Graph()
        self._bind_namespaces()
        self.triple_count = 0
        self.entity_counts = Counter()

    def _bind_namespaces(self):
        """Bind namespace prefixes."""
        self.g.bind("mlkg", MLKG)
        self.g.bind("mlkg_data", MLKG_DATA)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("owl", OWL)
        self.g.bind("xsd", XSD)

    def _add(self, s, p, o):
        """Add a triple and count it."""
        self.g.add((s, p, o))
        self.triple_count += 1

    def load_ontology(self, ontology_path):
        """Load the ontology schema into the graph."""
        self.g.parse(ontology_path, format="turtle")
        print(f"Loaded ontology from {ontology_path}")

    def add_publication(self, paper_data):
        """Add a publication entity and its datatype properties."""
        pid = paper_data.get("paperId")
        title = paper_data.get("title")
        if not pid or not title:
            return None

        uri = make_uri("publication", pid)

        # Type
        self._add(uri, RDF.type, MLKG.Publication)
        self.entity_counts["Publication"] += 1

        # Datatype properties
        self._add(uri, MLKG.title, Literal(title, datatype=XSD.string))

        abstract = paper_data.get("abstract")
        if abstract:
            self._add(uri, MLKG.abstract, Literal(abstract, datatype=XSD.string))

        year = paper_data.get("year")
        if year:
            self._add(uri, MLKG.publicationYear, Literal(int(year), datatype=XSD.integer))

        cc = paper_data.get("citationCount")
        if cc is not None:
            self._add(uri, MLKG.citationCount, Literal(int(cc), datatype=XSD.integer))

        # Label for readability
        self._add(uri, RDFS.label, Literal(title))

        return uri

    def add_author(self, author_data):
        """Add an author entity."""
        aid = author_data.get("authorId")
        name = author_data.get("name")
        if not aid or not name:
            return None

        uri = make_uri("author", aid)

        self._add(uri, RDF.type, MLKG.Author)
        self._add(uri, RDFS.label, Literal(name))
        self.entity_counts["Author"] += 1

        return uri

    def add_institution(self, name):
        """Add an institution entity."""
        if not name or name.strip() == "":
            return None

        uri = make_hash_uri("institution", name)

        # Only add if not already present
        if (uri, RDF.type, MLKG.Institution) not in self.g:
            self._add(uri, RDF.type, MLKG.Institution)
            self._add(uri, RDFS.label, Literal(name.strip()))
            self.entity_counts["Institution"] += 1

        return uri

    def add_venue(self, venue_name):
        """Add a venue entity."""
        if not venue_name or venue_name.strip() == "":
            return None

        uri = make_hash_uri("venue", venue_name)

        if (uri, RDF.type, MLKG.Venue) not in self.g:
            self._add(uri, RDF.type, MLKG.Venue)
            self._add(uri, RDFS.label, Literal(venue_name.strip()))
            self.entity_counts["Venue"] += 1

        return uri

    def add_research_area(self, area_id):
        """Add a research area entity."""
        if area_id not in RESEARCH_AREAS:
            return None

        uri = make_hash_uri("area", area_id)
        info = RESEARCH_AREAS[area_id]

        if (uri, RDF.type, MLKG.ResearchArea) not in self.g:
            self._add(uri, RDF.type, MLKG.ResearchArea)
            self._add(uri, RDFS.label, Literal(info["label"]))
            self.entity_counts["ResearchArea"] += 1

        return uri

    def add_research_topic(self, topic_id):
        """Add a research topic entity and link to its area."""
        if topic_id not in RESEARCH_TOPICS:
            return None

        uri = make_hash_uri("topic", topic_id)
        info = RESEARCH_TOPICS[topic_id]

        if (uri, RDF.type, MLKG.ResearchTopic) not in self.g:
            self._add(uri, RDF.type, MLKG.ResearchTopic)
            self._add(uri, RDFS.label, Literal(info["label"]))
            self.entity_counts["ResearchTopic"] += 1

            # Link topic to its area
            area_uri = self.add_research_area(info["area"])
            if area_uri:
                self._add(uri, MLKG.topicInArea, area_uri)

        return uri

    def add_dataset(self, name):
        """Add a dataset entity."""
        if not name or name.strip() == "":
            return None

        uri = make_hash_uri("dataset", name)

        if (uri, RDF.type, MLKG.Dataset) not in self.g:
            self._add(uri, RDF.type, MLKG.Dataset)
            self._add(uri, RDFS.label, Literal(name.strip()))
            self.entity_counts["Dataset"] += 1

        return uri

    def add_code_repository(self, url, paper_uri=None):
        """Add a code repository entity."""
        if not url:
            return None

        uri = make_hash_uri("code", url)

        if (uri, RDF.type, MLKG.CodeRepository) not in self.g:
            self._add(uri, RDF.type, MLKG.CodeRepository)
            self._add(uri, RDFS.label, Literal(url))
            self.entity_counts["CodeRepository"] += 1

            if paper_uri:
                self._add(uri, MLKG.implementationOf, paper_uri)

        return uri

    def build_from_extracted_data(self, papers_file, authors_file=None):
        """
        Build the complete knowledge graph from extracted JSON data.

        This is the main entry point that processes all papers and
        creates all entities and relationships.
        """
        print("=" * 60)
        print("BUILDING KNOWLEDGE GRAPH")
        print("=" * 60)

        # Load data
        with open(papers_file) as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {papers_file}")

        authors_data = {}
        if authors_file and Path(authors_file).exists():
            with open(authors_file) as f:
                authors_data = json.load(f)
            print(f"Loaded {len(authors_data)} authors from {authors_file}")

        # Track URIs for relationship building
        paper_uris = {}   # paperId -> URI
        author_uris = {}  # authorId -> URI

        # ---------------------------------------------------
        # Phase 1: Add publications
        # ---------------------------------------------------
        print("\nPhase 1: Adding publications...")
        for pid, pdata in papers.items():
            uri = self.add_publication(pdata)
            if uri:
                paper_uris[pid] = uri

        # ---------------------------------------------------
        # Phase 2: Add authors and author-paper relationships
        # ---------------------------------------------------
        print("Phase 2: Adding authors and authorship links...")
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            if not pub_uri:
                continue

            authors_list = pdata.get("authors", [])
            paper_author_uris = []

            for i, auth in enumerate(authors_list):
                aid = auth.get("authorId")
                name = auth.get("name")
                if not aid or not name:
                    continue

                # Add author
                author_uri = make_uri("author", aid)
                if aid not in author_uris:
                    # Use detailed author data if available
                    if aid in authors_data:
                        self.add_author(authors_data[aid])
                    else:
                        self.add_author(auth)
                    author_uris[aid] = author_uri

                    # Add affiliations
                    affiliations = authors_data.get(aid, {}).get("affiliations", [])
                    if not affiliations:
                        affiliations = auth.get("affiliations", [])
                    for aff in (affiliations or []):
                        if aff:
                            inst_uri = self.add_institution(aff)
                            if inst_uri:
                                self._add(author_uri, MLKG.affiliatedWith, inst_uri)

                # authorOf relationship
                self._add(author_uri, MLKG.authorOf, pub_uri)

                # First author
                if i == 0:
                    self._add(pub_uri, MLKG.firstAuthor, author_uri)

                paper_author_uris.append(author_uri)

            # coauthorWith relationships (symmetric, but only add once per pair)
            for i in range(len(paper_author_uris)):
                for j in range(i + 1, len(paper_author_uris)):
                    self._add(paper_author_uris[i], MLKG.coauthorWith,
                              paper_author_uris[j])

        # ---------------------------------------------------
        # Phase 3: Add venues and publishedIn
        # ---------------------------------------------------
        print("Phase 3: Adding venues...")
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            venue_name = pdata.get("venue")
            if pub_uri and venue_name:
                venue_uri = self.add_venue(venue_name)
                if venue_uri:
                    self._add(pub_uri, MLKG.publishedIn, venue_uri)

        # ---------------------------------------------------
        # Phase 4: Add citation links
        # ---------------------------------------------------
        print("Phase 4: Adding citation links...")
        citation_count = 0
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            if not pub_uri:
                continue

            # References (papers this paper cites)
            for ref in pdata.get("references", []):
                if isinstance(ref, dict):
                    ref_id = ref.get("paperId")
                else:
                    ref_id = ref

                if ref_id and ref_id in paper_uris:
                    self._add(pub_uri, MLKG.cites, paper_uris[ref_id])
                    citation_count += 1

            # Citations (papers that cite this paper)
            for cit in pdata.get("citations", []):
                if isinstance(cit, dict):
                    cit_id = cit.get("paperId")
                else:
                    cit_id = cit

                if cit_id and cit_id in paper_uris:
                    self._add(paper_uris[cit_id], MLKG.cites, pub_uri)
                    citation_count += 1

        print(f"  Added {citation_count} citation links")

        # ---------------------------------------------------
        # Phase 5: Topic and area classification
        # ---------------------------------------------------
        print("Phase 5: Classifying topics and areas...")
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            if not pub_uri:
                continue

            # Combine title + abstract for matching
            text = (pdata.get("title", "") + " " +
                    (pdata.get("abstract", "") or ""))

            # Match topics
            topics = match_topics(text)
            for tid in topics:
                topic_uri = self.add_research_topic(tid)
                if topic_uri:
                    self._add(pub_uri, MLKG.hasKeyword, topic_uri)

            # Match areas
            areas = match_areas(text, topics)
            for area_id in areas:
                area_uri = self.add_research_area(area_id)
                if area_uri:
                    self._add(pub_uri, MLKG.inArea, area_uri)

            # Also use S2's fieldsOfStudy if available
            for field in pdata.get("fieldsOfStudy", []) or []:
                if field:
                    area_uri = make_hash_uri("area", field)
                    if (area_uri, RDF.type, MLKG.ResearchArea) not in self.g:
                        self._add(area_uri, RDF.type, MLKG.ResearchArea)
                        self._add(area_uri, RDFS.label, Literal(field))
                        self.entity_counts["ResearchArea"] += 1
                    self._add(pub_uri, MLKG.inArea, area_uri)

        # ---------------------------------------------------
        # Phase 6: Code repositories (from PwC)
        # ---------------------------------------------------
        print("Phase 6: Adding code repositories...")
        code_count = 0
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            if not pub_uri:
                continue

            for repo_url in pdata.get("code_repositories", []) or []:
                if repo_url:
                    code_uri = self.add_code_repository(repo_url, pub_uri)
                    if code_uri:
                        self._add(pub_uri, MLKG.hasCode, code_uri)
                        code_count += 1
        print(f"  Added {code_count} code repository links")

        # ---------------------------------------------------
        # Phase 7: PwC tasks as research topics
        # ---------------------------------------------------
        print("Phase 7: Adding PwC tasks as topics...")
        pwc_topic_count = 0
        for pid, pdata in papers.items():
            pub_uri = paper_uris.get(pid)
            if not pub_uri:
                continue

            for task in pdata.get("pwc_tasks", []) or []:
                if task:
                    topic_uri = make_hash_uri("topic", task)
                    if (topic_uri, RDF.type, MLKG.ResearchTopic) not in self.g:
                        self._add(topic_uri, RDF.type, MLKG.ResearchTopic)
                        self._add(topic_uri, RDFS.label, Literal(task))
                        self.entity_counts["ResearchTopic"] += 1
                    self._add(pub_uri, MLKG.hasKeyword, topic_uri)
                    pwc_topic_count += 1
        print(f"  Added {pwc_topic_count} PwC task-topic links")

        # ---------------------------------------------------
        # Summary
        # ---------------------------------------------------
        print(f"\n{'=' * 60}")
        print("KNOWLEDGE GRAPH BUILD COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total triples: {len(self.g)}")
        print(f"\nEntity counts:")
        for etype, count in sorted(self.entity_counts.items()):
            print(f"  {etype}: {count}")

        return self.g

    def save_graph(self, output_path, fmt="turtle"):
        """Save the graph to a file."""
        self.g.serialize(destination=output_path, format=fmt)
        print(f"\nSaved graph ({len(self.g)} triples) to {output_path}")

    def load_graph(self, input_path, fmt="turtle"):
        """Load a graph from file."""
        self.g.parse(input_path, format=fmt)
        print(f"Loaded graph ({len(self.g)} triples) from {input_path}")

    def get_stats(self):
        """Print detailed graph statistics."""
        print(f"\n{'=' * 60}")
        print("KNOWLEDGE GRAPH STATISTICS")
        print(f"{'=' * 60}")
        print(f"Total triples: {len(self.g)}")

        # Count entities by type
        classes = [
            ("Publication", MLKG.Publication),
            ("Author", MLKG.Author),
            ("Institution", MLKG.Institution),
            ("Venue", MLKG.Venue),
            ("ResearchArea", MLKG.ResearchArea),
            ("ResearchTopic", MLKG.ResearchTopic),
            ("Dataset", MLKG.Dataset),
            ("CodeRepository", MLKG.CodeRepository),
        ]

        print("\nEntities:")
        for name, cls in classes:
            count = len(list(self.g.subjects(RDF.type, cls)))
            print(f"  {name}: {count}")

        # Count relationships
        properties = [
            ("authorOf", MLKG.authorOf),
            ("cites", MLKG.cites),
            ("publishedIn", MLKG.publishedIn),
            ("hasKeyword", MLKG.hasKeyword),
            ("inArea", MLKG.inArea),
            ("affiliatedWith", MLKG.affiliatedWith),
            ("coauthorWith", MLKG.coauthorWith),
            ("firstAuthor", MLKG.firstAuthor),
            ("topicInArea", MLKG.topicInArea),
        ]

        print("\nRelationships:")
        for name, prop in properties:
            count = len(list(self.g.subject_objects(prop)))
            print(f"  {name}: {count}")


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()

    # Load ontology
    builder.load_ontology("schema/ontology.ttl")

    # Build from extracted data
    builder.build_from_extracted_data(
        papers_file="data/raw/s2_papers.json",
        authors_file="data/raw/s2_authors.json"
    )

    # Save
    builder.save_graph("data/processed/ml_research_kg.ttl")

    # Print stats
    builder.get_stats()
