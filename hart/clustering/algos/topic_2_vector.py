import umap
from typing import List
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import plotly.express as px


print("loading topic2vec ....")
from top2vec import Top2Vec
# import hdbscan
import seaborn as sns


class Topic2VecClustering:

    def __init__(self, **kwargs):
        self.ngram_vocab = kwargs.get('ngram_vocab', True)
        self.contextual_top2vec = kwargs.get('contextual_top2vec', False)
        self.gpu_hdbscan = kwargs.get('gpu_hdbscan', True)
        self.gpu_umap = kwargs.get('gpu_umap', True)
        self.model = None
        self.model_name = kwargs.get('name', str(uuid4()))

    def init_model(self, prompts: List[str]):
        self.model = Top2Vec(documents=prompts,
                             ngram_vocab=self.ngram_vocab,
                             contextual_top2vec=self.contextual_top2vec,
                             gpu_hdbscan=self.gpu_hdbscan,
                             gpu_umap=self.gpu_umap)
        return self.model

    def get_model(self):
        if self.model is not None:
            return self.model

        raise ("Model not Initialized")

    def load_model(self, model_file_name: str):
        self.model = Top2Vec.load(model_file_name)
        return self.model

    def save_model(self, model_file_name: str):
        m = self.get_model()
        m.save(model_file_name)

    def get_save_path(self, file_name, save_path):
        file_name = self.model_name + "_" + file_name
        save_path = f"{save_path}{file_name}"
        return save_path

    def plot_topic_distribution(self, file_name="model_distribution", save_path='./', show_fig=True):
        save_path = self.get_save_path(file_name, save_path) + '.png'
        topic_sizes, topic_nums = self.model.get_topic_sizes()

        sorted_indices = np.argsort(topic_sizes)[::-1]
        topic_sizes = topic_sizes[sorted_indices]
        topic_nums = topic_nums[sorted_indices]

        # Plot
        plt.figure(figsize=(14, 7))
        plt.bar(range(len(topic_sizes)), topic_sizes,
                color=plt.cm.viridis(np.linspace(0, 1, len(topic_sizes))))
        plt.title("Topic Sizes by Topic Number", fontsize=16, fontweight="bold")
        plt.xlabel("Topics (sorted by size)", fontsize=12)
        plt.ylabel("Number of Documents / Tokens", fontsize=12)
        plt.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.save_fig(save_path)

        if show_fig:
            plt.show()

    def get_umap_document_embeddings(self, nearest_neighbours=15, dimensions=2, min_dist=0.05, metric="cosine"):
        # Get document embeddings (high dim)
        doc_embeds = self.model.document_vectors  # shape: (num_docs, embed_dim)

        # (Optional) If embeddings are not yet reduced, reduce them to 2D
        # Top2Vec sometimes already applies UMAP internally; but you can override / re-reduce:
        umap_reducer = umap.UMAP(n_neighbors=nearest_neighbours, n_components=dimensions, min_dist=min_dist,
                                 metric=metric, random_state=42)
        embeds = umap_reducer.fit_transform(doc_embeds)  # shape: (num_docs, 2)
        return embeds

    def _get_cluster_names(self):
        num_topics = self.model.get_num_topics()
        topic_words_dict = {}

        topics, scores, topic_nums = self.model.get_topics()

        for words, topic_num in zip(topics, topic_nums):
            # Use top 3 words as topic name
            topic_name = f"T{topic_num}: {', '.join(words[:3])}"
            topic_words_dict[topic_num] = topic_name

        # Add label for un-clustered
        topic_words_dict[-1] = "Un-clustered"
        doc_topics = self.model.doc_top

        cluster_names = [topic_words_dict.get(t, f"Topic {t}") for t in doc_topics]
        return cluster_names

    def plot_static_clusters(self, file_name="model_clusters", save_path='./', show_fig=True):
        # Create the plot
        save_path = self.get_save_path(file_name, save_path) + '.png'
        # Create labels for plotting
        labels = self.model.doc_top
        label_names = self._get_cluster_names()

        plt.figure(figsize=(15, 15))

        # Count documents per cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of unclustered documents: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

        # Create color palette
        n_colors = len(unique_labels)
        if -1 in unique_labels:
            # Gray for noise points
            colors = ['gray'] + list(sns.color_palette("Set2", n_colors - 1))
            palette = {label: colors[i] for i, label in enumerate(sorted(unique_labels))}
        else:
            palette = sns.color_palette("Set2", np.unique(labels).size)

        # Plot
        embeds_2d = self.get_umap_document_embeddings()
        scatter = sns.scatterplot(
            x=embeds_2d[:, 0],
            y=embeds_2d[:, 1],
            hue=label_names,
            palette=palette if isinstance(palette, dict) else None,
            s=15,
            alpha=0.7,
            edgecolor='none'
        )

        plt.title(f"Top2Vec Document Embeddings: {n_clusters} Topic Clusters", fontsize=14, fontweight='bold')
        plt.xlabel("UMAP dimension 1")
        plt.ylabel("UMAP dimension 2")

        # Adjust legend
        plt.legend(
            title="Topics",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=1 if n_clusters < 15 else 2,
            fontsize=9
        )

        plt.tight_layout()
        plt.savefig(save_path)
        if show_fig:
            plt.show()

    def plot_interactive_clusters_with_legend(self, file_name="model_clusters_interactive", save_path='./',
                                              show_fig=True):
        """Create interactive 3D plot with legend for each topic"""

        save_path = self.get_save_path(file_name, save_path) + '.html'

        # Get data
        embeds_3d = self.get_umap_document_embeddings(dimensions=3)
        labels = self.model.doc_top
        label_names = self._get_cluster_names()

        # Get unique topics and their info
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of unclustered documents: {n_noise}")

        # Create color palette
        n_colors = len(unique_labels)
        if -1 in unique_labels:
            colors_list = ['#808080'] + sns.color_palette("Set2", n_colors - 1).as_hex()
            color_map = {label: colors_list[i] for i, label in enumerate(sorted(unique_labels))}
        else:
            palette = sns.color_palette("Set2", n_colors).as_hex()
            color_map = {label: palette[i] for i, label in enumerate(sorted(unique_labels))}

        # Create figure
        fig = go.Figure()

        # Add a trace for each topic
        for topic_id in sorted(unique_labels):
            mask = labels == topic_id
            topic_embeds = embeds_3d[mask]
            topic_label_names = [label_names[i] for i in range(len(labels)) if labels[i] == topic_id]

            # Get topic name for legend
            topic_name = topic_label_names[0] if topic_label_names else f"Topic {topic_id}"

            fig.add_trace(go.Scatter3d(
                x=topic_embeds[:, 0],
                y=topic_embeds[:, 1],
                z=topic_embeds[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map[topic_id],
                    opacity=0.7,
                    line=dict(width=0)
                ),
                name=topic_name,
                text=[f"Doc {i}: {name}" for i, name in enumerate(topic_label_names)],
                hoverinfo='text'
            ))

        # Update layout
        fig.update_layout(
            title=f"Top2Vec Document Embeddings: {n_clusters} Topic Clusters (Interactive)",
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1200,
            height=800,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                font=dict(size=9)
            )
        )

        fig.write_html(save_path)
        print(f"Interactive plot saved to: {save_path}")

        if show_fig:
            fig.show()

        return fig

    def plot_interactive_with_centroids(self, file_name="clusters_with_centroids", save_path='./', show_fig=True):
        """Simplified version - just documents and centroids"""

        save_path = self.get_save_path(file_name, save_path) + '.html'

        embeds_3d = self.get_umap_document_embeddings(dimensions=3)
        labels = self.model.doc_top
        label_names = self._get_cluster_names()

        try:
            documents = self.model.documents
        except:
            documents = [f"Document {i}" for i in range(len(labels))]

        unique_labels = np.unique(labels)
        n_colors = len(unique_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of unclustered documents: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

        if -1 in unique_labels:
            colors_list = ['#808080'] + sns.color_palette("Set2", n_colors - 1).as_hex()
            color_map = {label: colors_list[i] for i, label in enumerate(sorted(unique_labels))}
        else:
            palette = sns.color_palette("Set2", n_colors).as_hex()
            color_map = {label: palette[i] for i, label in enumerate(sorted(unique_labels))}

        fig = go.Figure()

        # Add documents and centroids
        for topic_id in sorted(unique_labels):
            mask = labels == topic_id
            topic_embeds = embeds_3d[mask]
            topic_indices = np.where(mask)[0]

            topic_label_names = [label_names[i] for i in topic_indices]
            topic_name = topic_label_names[0] if topic_label_names else f"Topic {topic_id}"

            # Hover text
            hover_texts = [
                f"<b>{label_names[idx]}</b><br>Doc {idx}<br><i>{documents[idx][:100]}...</i>"
                for idx in topic_indices
            ]

            # Documents
            fig.add_trace(go.Scatter3d(
                x=topic_embeds[:, 0],
                y=topic_embeds[:, 1],
                z=topic_embeds[:, 2],
                mode='markers',
                marker=dict(size=5, color=color_map[topic_id], opacity=0.6),
                name=topic_name,
                text=hover_texts,
                hoverinfo='text'
            ))

            # Centroid
            if len(topic_embeds) > 0:
                centroid = np.mean(topic_embeds, axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode='markers',
                    marker=dict(size=15, color=color_map[topic_id], symbol='diamond',
                                line=dict(width=2, color='black')),
                    name=f'{topic_name} Centroid',
                    showlegend=False,
                    text=[f"<b>Centroid</b><br>{topic_name}"],
                    hoverinfo='text'
                ))

        fig.update_layout(
            title="Top2Vec Clusters with Centroids",
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1200,
            height=900
        )

        fig.write_html(save_path)
        if show_fig:
            fig.show()

        return fig