"""
Graph Embeddings and Link Prediction
Uses PyKEEN for training knowledge graph embeddings (TransE, RotatE, ComplEx).
Includes evaluation, visualization (t-SNE/PCA), and recommendation support.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from rdflib import Graph as RDFGraph, Namespace, RDF, RDFS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

MLKG = Namespace("http://example.org/mlkg/")
MLKG_DATA = Namespace("http://example.org/mlkg/data/")


class KGEmbeddingPipeline:
    """
    Train graph embeddings and run link prediction on the ML Research KG.
    """

    def __init__(self, kg_path, output_dir="output"):
        self.kg_path = kg_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rdf_graph = None
        self.triples = []
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        self.entity_labels = {}  # URI -> human label
        self.entity_types = {}   # URI -> class type
        self.model = None
        self.embeddings = None

    def load_kg(self):
        """Load the RDF knowledge graph and extract triples."""
        print("Loading knowledge graph...")
        self.rdf_graph = RDFGraph()
        self.rdf_graph.parse(self.kg_path, format="turtle")
        print(f"  Loaded {len(self.rdf_graph)} total RDF triples")

        # Extract entity labels and types
        for s, _, o in self.rdf_graph.triples((None, RDFS.label, None)):
            self.entity_labels[str(s)] = str(o)

        class_map = {
            str(MLKG.Publication): "Publication",
            str(MLKG.Author): "Author",
            str(MLKG.Institution): "Institution",
            str(MLKG.Venue): "Venue",
            str(MLKG.ResearchArea): "ResearchArea",
            str(MLKG.ResearchTopic): "ResearchTopic",
            str(MLKG.Dataset): "Dataset",
            str(MLKG.CodeRepository): "CodeRepository",
        }

        for s, _, o in self.rdf_graph.triples((None, RDF.type, None)):
            cls = str(o)
            if cls in class_map:
                self.entity_types[str(s)] = class_map[cls]

        # Extract object property triples (skip datatype properties and schema triples)
        skip_predicates = {
            str(RDF.type), str(RDFS.label), str(RDFS.comment),
            str(RDFS.domain), str(RDFS.range), str(RDFS.subClassOf),
            str(MLKG.title), str(MLKG.abstract),
            str(MLKG.publicationYear), str(MLKG.citationCount),
        }
        # Also skip OWL schema triples
        from rdflib import OWL
        skip_predicates.update({
            str(OWL.Class), str(OWL.ObjectProperty),
            str(OWL.DatatypeProperty), str(OWL.Ontology),
            str(OWL.SymmetricProperty),
        })

        for s, p, o in self.rdf_graph:
            s_str, p_str, o_str = str(s), str(p), str(o)

            # Skip non-object-property triples
            if p_str in skip_predicates:
                continue
            # Skip if object is a literal
            if not o_str.startswith("http"):
                continue
            # Skip schema-level triples
            if "www.w3.org" in s_str or "www.w3.org" in o_str:
                continue

            self.triples.append((s_str, p_str, o_str))

        # Build entity/relation mappings
        entities = set()
        relations = set()
        for s, p, o in self.triples:
            entities.add(s)
            entities.add(o)
            relations.add(p)

        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        print(f"  Object property triples: {len(self.triples)}")
        print(f"  Unique entities: {len(self.entity_to_id)}")
        print(f"  Unique relations: {len(self.relation_to_id)}")

        # Print relation breakdown
        rel_counts = defaultdict(int)
        for _, p, _ in self.triples:
            short_name = p.split("/")[-1]
            rel_counts[short_name] += 1
        print("\n  Relations breakdown:")
        for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
            print(f"    {rel}: {count}")

    def prepare_pykeen_triples(self):
        """Convert triples to PyKEEN TriplesFactory format."""
        try:
            from pykeen.triples import TriplesFactory
        except ImportError:
            print("PyKEEN not installed. Install with: pip install pykeen")
            return None

        # Convert to numpy array of strings
        triple_array = np.array(self.triples, dtype=str)

        tf = TriplesFactory.from_labeled_triples(triple_array)
        print(f"\n  PyKEEN TriplesFactory: {tf.num_triples} triples, "
              f"{tf.num_entities} entities, {tf.num_relations} relations")

        return tf

    def train_embeddings(self, model_name="TransE", embedding_dim=128,
                          num_epochs=100, lr=0.01, test_ratio=0.1,
                          val_ratio=0.1):
        """
        Train knowledge graph embeddings using PyKEEN.

        Args:
            model_name: One of 'TransE', 'RotatE', 'ComplEx'
            embedding_dim: Dimension of entity/relation embeddings
            num_epochs: Training epochs
            lr: Learning rate
            test_ratio: Fraction of triples held out for testing
            val_ratio: Fraction of triples held out for validation
        """
        try:
            from pykeen.pipeline import pipeline
        except ImportError:
            print("PyKEEN not installed. Using fallback embedding method.")
            self._train_fallback_embeddings(embedding_dim)
            return

        print(f"\nTraining {model_name} embeddings...")
        print(f"  Dimensions: {embedding_dim}")
        print(f"  Epochs: {num_epochs}")

        tf = self.prepare_pykeen_triples()
        if tf is None:
            self._train_fallback_embeddings(embedding_dim)
            return

        # Split data
        training, testing, validation = tf.split([
            1.0 - test_ratio - val_ratio,
            test_ratio,
            val_ratio
        ])

        # Train
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model_name,
            model_kwargs={"embedding_dim": embedding_dim},
            optimizer="Adam",
            optimizer_kwargs={"lr": lr},
            training_kwargs={
                "num_epochs": num_epochs,
                "batch_size": min(256, len(self.triples) // 4 + 1),
            },
            evaluation_kwargs={"batch_size": 256},
            random_seed=42,
        )

        self.model = result.model
        self.training_result = result

        # Extract embeddings
        entity_embeddings = (
            result.model.entity_representations[0]()
            .detach()
            .cpu()
            .numpy()
        )

        # Map back to our entity IDs
        entity_id_map = training.entity_to_id
        self.embeddings = {}
        for entity_str, idx in entity_id_map.items():
            if idx < len(entity_embeddings):
                self.embeddings[entity_str] = entity_embeddings[idx]

        print(f"\n  Extracted embeddings for {len(self.embeddings)} entities")

        # Print evaluation metrics
        metrics = result.metric_results.to_dict()
        print(f"\n  Evaluation Metrics:")
        for key in ['mean_reciprocal_rank', 'hits_at_1', 'hits_at_3', 'hits_at_10']:
            both_key = f"both.realistic.{key}"
            if both_key in metrics:
                print(f"    {key}: {metrics[both_key]:.4f}")

        # Save metrics
        self._save_metrics(metrics)

        return result

    def _train_fallback_embeddings(self, embedding_dim=64):
        """
        Fallback: train simple embeddings without PyKEEN.
        Uses adjacency-based node2vec-style random embeddings
        refined by SVD on the adjacency matrix.
        """
        print("\nUsing fallback SVD-based embeddings...")

        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds

        n_entities = len(self.entity_to_id)
        adj = lil_matrix((n_entities, n_entities), dtype=np.float32)

        for s, p, o in self.triples:
            si = self.entity_to_id[s]
            oi = self.entity_to_id[o]
            adj[si, oi] = 1.0
            adj[oi, si] = 1.0  # undirected

        # SVD
        k = min(embedding_dim, n_entities - 2)
        U, S, Vt = svds(adj.tocsr(), k=k)

        # Entity embeddings = U * sqrt(S)
        emb_matrix = U * np.sqrt(S)

        self.embeddings = {}
        for entity, idx in self.entity_to_id.items():
            self.embeddings[entity] = emb_matrix[idx]

        print(f"  Generated {len(self.embeddings)} embeddings of dim {k}")

    def _save_metrics(self, metrics):
        """Save evaluation metrics to JSON."""
        output_path = self.output_dir / "link_prediction_metrics.json"
        # Filter to key metrics
        key_metrics = {}
        for key in ['mean_reciprocal_rank', 'hits_at_1', 'hits_at_3',
                     'hits_at_10', 'mean_rank']:
            for prefix in ['both.realistic.', 'head.realistic.', 'tail.realistic.']:
                full_key = prefix + key
                if full_key in metrics:
                    key_metrics[full_key] = float(metrics[full_key])

        with open(output_path, "w") as f:
            json.dump(key_metrics, f, indent=2)
        print(f"  Saved metrics to {output_path}")

    def visualize_embeddings(self, method="tsne", perplexity=30,
                              max_entities=500, figsize=(14, 10)):
        """
        Visualize entity embeddings using t-SNE or PCA.
        Color-coded by entity type.
        """
        if not self.embeddings:
            print("No embeddings to visualize. Train first.")
            return

        print(f"\nVisualizing embeddings with {method.upper()}...")

        # Collect entities with embeddings and types
        entities = []
        vectors = []
        types = []
        labels = []

        for entity, emb in list(self.embeddings.items())[:max_entities]:
            etype = self.entity_types.get(entity, "Other")
            label = self.entity_labels.get(entity, entity.split("/")[-1])

            entities.append(entity)
            vectors.append(emb)
            types.append(etype)
            labels.append(label[:40])  # Truncate

        X = np.array(vectors)

        # Reduce dimensions
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(perplexity, len(X) - 1),
                           random_state=42, max_iter=1000)
        else:
            reducer = PCA(n_components=2, random_state=42)

        X_2d = reducer.fit_transform(X)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        type_colors = {
            "Publication": "#2E5E8A",
            "Author": "#D97706",
            "Institution": "#059669",
            "Venue": "#DC2626",
            "ResearchArea": "#7C3AED",
            "ResearchTopic": "#DB2777",
            "Dataset": "#0891B2",
            "CodeRepository": "#65A30D",
            "Other": "#6B7280",
        }

        # Plot each type
        for etype in sorted(set(types)):
            mask = [t == etype for t in types]
            indices = [i for i, m in enumerate(mask) if m]
            if not indices:
                continue
            ax.scatter(
                X_2d[indices, 0], X_2d[indices, 1],
                c=type_colors.get(etype, "#6B7280"),
                label=f"{etype} ({len(indices)})",
                alpha=0.6, s=30, edgecolors='white', linewidth=0.3
            )

        ax.set_title(f"ML Research KG Entity Embeddings ({method.upper()})",
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")
        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"embeddings_{method}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved visualization to {output_path}")

        return X_2d, types, labels

    def predict_links(self, head_entity=None, relation=None,
                       top_k=10):
        """
        Predict missing links using trained embeddings.
        Uses cosine similarity for entity similarity scoring.
        """
        if not self.embeddings:
            print("No embeddings available. Train first.")
            return []

        print(f"\nPredicting links (top {top_k})...")

        if head_entity and head_entity in self.embeddings:
            # Find most similar entities
            head_emb = self.embeddings[head_entity]
            scores = []

            for entity, emb in self.embeddings.items():
                if entity == head_entity:
                    continue
                # Cosine similarity
                cos_sim = np.dot(head_emb, emb) / (
                    np.linalg.norm(head_emb) * np.linalg.norm(emb) + 1e-8
                )
                scores.append((entity, cos_sim))

            scores.sort(key=lambda x: -x[1])
            top_predictions = scores[:top_k]

            print(f"\n  Top {top_k} similar entities to "
                  f"'{self.entity_labels.get(head_entity, head_entity)}':")
            for entity, score in top_predictions:
                label = self.entity_labels.get(entity, entity.split("/")[-1])
                etype = self.entity_types.get(entity, "?")
                print(f"    {score:.4f} - [{etype}] {label}")

            return top_predictions

        else:
            # General link prediction: find entity pairs most likely to be linked
            # Sample random pairs and score them
            entity_list = list(self.embeddings.keys())
            existing_links = set((s, o) for s, _, o in self.triples)

            scores = []
            np.random.seed(42)

            # Score a sample of non-existing pairs
            n_samples = min(10000, len(entity_list) ** 2 // 10)
            for _ in range(n_samples):
                i = np.random.randint(len(entity_list))
                j = np.random.randint(len(entity_list))
                if i == j:
                    continue

                e1, e2 = entity_list[i], entity_list[j]
                if (e1, e2) in existing_links:
                    continue

                emb1, emb2 = self.embeddings[e1], self.embeddings[e2]
                cos_sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
                )
                scores.append((e1, e2, cos_sim))

            scores.sort(key=lambda x: -x[2])
            top_predictions = scores[:top_k]

            print(f"\n  Top {top_k} predicted new links:")
            for e1, e2, score in top_predictions:
                l1 = self.entity_labels.get(e1, e1.split("/")[-1])
                l2 = self.entity_labels.get(e2, e2.split("/")[-1])
                t1 = self.entity_types.get(e1, "?")
                t2 = self.entity_types.get(e2, "?")
                print(f"    {score:.4f} - [{t1}] {l1[:30]}  <-->  [{t2}] {l2[:30]}")

            return top_predictions

    def evaluate_link_prediction(self, test_ratio=0.1):
        """
        Evaluate link prediction using ranking metrics.
        Holds out a fraction of triples and tries to predict them.
        """
        if not self.embeddings:
            print("No embeddings available.")
            return {}

        print("\nEvaluating link prediction...")

        # Split triples
        np.random.seed(42)
        indices = np.random.permutation(len(self.triples))
        n_test = max(1, int(len(self.triples) * test_ratio))
        test_indices = indices[:n_test]
        test_triples = [self.triples[i] for i in test_indices]

        entity_list = list(self.embeddings.keys())
        ranks = []

        for s, p, o in test_triples:
            if s not in self.embeddings or o not in self.embeddings:
                continue

            head_emb = self.embeddings[s]
            true_emb = self.embeddings[o]
            true_score = np.dot(head_emb, true_emb) / (
                np.linalg.norm(head_emb) * np.linalg.norm(true_emb) + 1e-8
            )

            # Rank against random negatives
            n_negatives = min(100, len(entity_list))
            neg_scores = []
            for _ in range(n_negatives):
                neg = entity_list[np.random.randint(len(entity_list))]
                if neg == o or neg not in self.embeddings:
                    continue
                neg_emb = self.embeddings[neg]
                neg_score = np.dot(head_emb, neg_emb) / (
                    np.linalg.norm(head_emb) * np.linalg.norm(neg_emb) + 1e-8
                )
                neg_scores.append(neg_score)

            rank = 1 + sum(1 for ns in neg_scores if ns >= true_score)
            ranks.append(rank)

        if not ranks:
            print("  No test triples could be evaluated.")
            return {}

        ranks = np.array(ranks)
        metrics = {
            "MRR": float(np.mean(1.0 / ranks)),
            "Hits@1": float(np.mean(ranks <= 1)),
            "Hits@3": float(np.mean(ranks <= 3)),
            "Hits@10": float(np.mean(ranks <= 10)),
            "Mean_Rank": float(np.mean(ranks)),
            "Num_Test_Triples": len(ranks),
        }

        print(f"\n  Link Prediction Results:")
        print(f"    MRR:      {metrics['MRR']:.4f}  (target: > 0.3)")
        print(f"    Hits@1:   {metrics['Hits@1']:.4f}")
        print(f"    Hits@3:   {metrics['Hits@3']:.4f}")
        print(f"    Hits@10:  {metrics['Hits@10']:.4f}  (target: > 0.5)")
        print(f"    Mean Rank: {metrics['Mean_Rank']:.1f}")
        print(f"    Test triples: {metrics['Num_Test_Triples']}")

        # Save
        output_path = self.output_dir / "link_prediction_results.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved results to {output_path}")

        return metrics

    def visualize_link_prediction_results(self, metrics, figsize=(8, 5)):
        """Create a bar chart of link prediction metrics."""
        if not metrics:
            return

        fig, ax = plt.subplots(figsize=figsize)

        metric_names = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
        values = [metrics.get(m, 0) for m in metric_names]
        targets = [0.3, None, None, 0.5]  # Target lines

        colors = ["#2E5E8A" if v >= (t or 0) else "#DC2626"
                  for v, t in zip(values, targets)]

        bars = ax.bar(metric_names, values, color=colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5)

        # Add target lines
        if targets[0]:
            ax.axhline(y=targets[0], color='#D97706', linestyle='--',
                       alpha=0.7, label=f'MRR target ({targets[0]})')
        if targets[3]:
            ax.axhline(y=targets[3], color='#D97706', linestyle=':',
                       alpha=0.7, label=f'Hits@10 target ({targets[3]})')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Link Prediction Performance", fontsize=14,
                     fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend()
        plt.tight_layout()

        output_path = self.output_dir / "link_prediction_chart.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved chart to {output_path}")

    def save_embeddings(self):
        """Save embeddings to file."""
        if not self.embeddings:
            return

        output_path = self.output_dir / "entity_embeddings.json"
        serializable = {
            k: v.tolist() for k, v in self.embeddings.items()
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f)
        print(f"  Saved {len(self.embeddings)} embeddings to {output_path}")


def run_full_pipeline(kg_path, output_dir="output", model_name="TransE",
                       embedding_dim=128, num_epochs=100):
    """Run the complete embedding + link prediction pipeline."""
    pipe = KGEmbeddingPipeline(kg_path, output_dir)

    # Load KG
    pipe.load_kg()

    # Train embeddings
    pipe.train_embeddings(
        model_name=model_name,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs
    )

    # Visualize
    pipe.visualize_embeddings(method="tsne")
    pipe.visualize_embeddings(method="pca")

    # Evaluate link prediction
    metrics = pipe.evaluate_link_prediction()

    # Visualize results
    pipe.visualize_link_prediction_results(metrics)

    # Save embeddings
    pipe.save_embeddings()

    # Example: predict links for a sample entity
    if pipe.embeddings:
        sample_entity = list(pipe.embeddings.keys())[0]
        pipe.predict_links(head_entity=sample_entity, top_k=5)

    return pipe


if __name__ == "__main__":
    run_full_pipeline(
        kg_path="data/processed/ml_research_kg.ttl",
        output_dir="output",
        model_name="TransE",
        embedding_dim=128,
        num_epochs=100
    )
