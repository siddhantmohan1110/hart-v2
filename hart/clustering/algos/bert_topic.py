"""
BERTopic Document Clustering with Interactive 3D Visualization
Author: PhD Student - Computer Science
Purpose: Demonstrate topic modeling, clustering, and visualization using BERTopic
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from bertopic import BERTopic
from cuml.manifold import UMAP

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from typing import Dict, Sequence
from tqdm import tqdm

import torch


import json
from time import time 
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import KMeans


from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"

# Create your representation model
# generator = pipeline('text2text-generation', model='google/flan-t5-base', device ='cuda')
# representation_model = TextGeneration(generator, prompt = prompt)
# Train BERTopic with a custom OnlineCountVectorizer
vectorizer_model = OnlineCountVectorizer(stop_words="english")


class BERTopicAnalyzer:
    """
    A class to perform topic modeling using BERTopic with visualization capabilities.
    """
    
    def __init__(self, clustering_model, documents=None, min_topic_size=10, n_components=3, embeddings = None):
        """
        Initialize the BERTopic analyzer.
        
        Parameters:
        -----------
        documents : list, optional
            List of text documents for analysis
        min_topic_size : int, default=10
            Minimum number of documents per topic
        n_components : int, default=3
            Number of dimensions for UMAP reduction (3 for 3D visualization)
        """
        self.documents = documents
        self.min_topic_size = min_topic_size
        self.n_components = n_components
        self.topic_model = None
        self.topics = None
        self.probabilities = None
        self.embeddings = None
        self.clustering_model = clustering_model
        # self.representation_model = representation_model

    @staticmethod
    def load_data(path):
        with open(path, 'r') as f: 
            meta_data = json.load(f)
    
        processed_data = []
        for id, value in meta_data.items():
            processed_data.append({"id": id, "prompt": value['prompt']})
    
        prompts = [p['prompt'] for p in processed_data]
        return prompts
        
    def create_custom_documents(self):
    #     """
    #     Create custom sample documents for demonstration.
    #     """
        self.documents = self.load_data()
        print(f"Created {len(self.documents)} custom documents")
        
    def fit_model(self, documents = None, embeddings = None):
        """
        Fit the BERTopic model to the documents.
        
        Parameters:
        -----------
        use_gpu : bool, default=False
            Whether to use GPU acceleration (requires CUDA)
        """
        print("Initializing BERTopic model...")
        
        # Configure UMAP for 3D visualization
        umap_model = UMAP(
            n_components=self.n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        

        hdbscan_model = self.clustering_model
        
        # Initialize BERTopic with custom UMAP
        self.topic_model = BERTopic(
            # representation_model=representation_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=self.min_topic_size,
            verbose=True,
            calculate_probabilities=True,
            vectorizer_model=vectorizer_model, 
            # nr_topics="auto"
        )

        if documents is None: 
            documents = self.documents

        if not documents: 
            raise Exception("Documents not init")
        
        print("Fitting model to documents...")
        self.topics, self.probabilities = self.topic_model.fit_transform(documents, embeddings)


        if embeddings is None: 
        # Get embeddings for visualization
            self.embeddings = self.topic_model._extract_embeddings(self.documents)

        else:
            self.embeddings = embeddings
        
        print(f"Model fitted successfully! Found {len(set(self.topics)) - 1} topics (excluding outliers)")
        
    def visualize_3d_interactive(self):
        """
        Create an interactive 3D visualization of document clusters.
        """
        if self.topic_model is None:
            raise ValueError("Model must be fitted before visualization. Call fit_model() first.")
        
        print("Creating 3D visualization...")
        
        # Reduce embeddings to 3D using the fitted UMAP model
        embeddings_3d = self.topic_model.umap_model.transform(self.embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'topic': self.topics,
            'document': [doc[:100] + '...' if len(doc) > 100 else doc for doc in self.documents]
        })
        
        # Get topic labels
        topic_labels = {topic: f"Topic {topic}: {', '.join([word for word, _ in words[:3]])}" 
                       for topic, words in self.topic_model.get_topics().items() if topic != -1}
        topic_labels[-1] = "Outliers"
        
        df['topic_label'] = df['topic'].map(topic_labels)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df, 
            x='x', 
            y='y', 
            z='z',
            color='topic_label',
            hover_data=['document'],
            title='3D Interactive Visualization of Document Topics',
            labels={'topic_label': 'Topic'},
            width=900,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.8))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3',
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.show()
        return fig
    
    def plot_topic_distribution(self):
        """
        Create a bar plot showing the distribution of topics vs number of documents.
        """
        if self.topics is None:
            raise ValueError("Model must be fitted before plotting. Call fit_model() first.")
        
        print("Creating topic distribution plot...")
        
        # Count documents per topic
        topic_counts = pd.Series(self.topics).value_counts().sort_index()
        
        # Get topic labels with top words
        topic_info = self.topic_model.get_topic_info()
        topic_labels = {}
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:
                # Extract top 3 words from the representation
                words = eval(row['Representation']) if isinstance(row['Representation'], str) else row['Representation']
                top_words = ', '.join(words[:3]) if isinstance(words, list) else str(words)[:30]
                topic_labels[row['Topic']] = f"T{row['Topic']}: {top_words}"
            else:
                topic_labels[row['Topic']] = "Outliers"
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'Topic': topic_counts.index,
            'Document Count': topic_counts.values,
            'Topic Label': [topic_labels.get(t, f"Topic {t}") for t in topic_counts.index]
        })
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_plot)))
        bars = ax1.bar(range(len(df_plot)), df_plot['Document Count'], color=colors)
        ax1.set_xlabel('Topic', fontsize=12)
        ax1.set_ylabel('Number of Documents', fontsize=12)
        ax1.set_title('Distribution of Documents across Topics', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(df_plot)))
        ax1.set_xticklabels([f"T{t}" if t != -1 else "Out" for t in df_plot['Topic']], rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        # Pie chart for proportion
        ax2.pie(df_plot['Document Count'], 
               labels=[f"T{t}" if t != -1 else "Outliers" for t in df_plot['Topic']],
               autopct='%1.1f%%',
               colors=colors,
               startangle=90)
        ax2.set_title('Proportion of Documents per Topic', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("TOPIC DISTRIBUTION SUMMARY")
        print("="*50)
        print(f"Total number of topics: {len(df_plot) - 1} (excluding outliers)")
        print(f"Total documents: {df_plot['Document Count'].sum()}")
        print(f"Average documents per topic: {df_plot[df_plot['Topic'] != -1]['Document Count'].mean():.2f}")
        print(f"Outlier documents: {df_plot[df_plot['Topic'] == -1]['Document Count'].values[0] if -1 in df_plot['Topic'].values else 0}")
        print("\nTop 5 largest topics:")
        for _, row in df_plot.nlargest(5, 'Document Count').iterrows():
            print(f"  {row['Topic Label']}: {row['Document Count']} documents")
        
        return fig
    
    def get_topic_words(self, n_words=10):
        """
        Display the top words for each topic.
        
        Parameters:
        -----------
        n_words : int, default=10
            Number of top words to display per topic
        """
        if self.topic_model is None:
            raise ValueError("Model must be fitted first. Call fit_model().")
        
        print("\n" + "="*50)
        print(f"TOP {n_words} WORDS PER TOPIC")
        print("="*50)
        
        for topic in sorted(set(self.topics)):
            if topic != -1:  # Skip outliers
                words = self.topic_model.get_topic(topic)[:n_words]
                word_list = ', '.join([word for word, _ in words])
                print(f"\nTopic {topic}:")
                print(f"  Keywords: {word_list}")


                
# # Main execution
# def main():
#     """
#     Main function to demonstrate BERTopic clustering and visualization.
#     """
#     # Initialize analyzer
#     analyzer = BERTopicAnalyzer(min_topic_size=5, n_components=3)
    
#     # Choose data source (comment/uncomment as needed)
#     # Option 1: Use custom documents
#     analyzer.create_custom_documents()
    
#     # Option 2: Load from 20 newsgroups (uncomment to use)
#     # analyzer.load_sample_data(n_samples=500)
    
#     # Option 3: Use your own documents (uncomment and modify)
#     analyzer.load_data()
#     # analyzer.documents = your_documents
    
#     # Fit the model
#     analyzer.fit_model()
    
#     # Display topic words
#     analyzer.get_topic_words(n_words=10)
    
#     # Create visualizations
#     analyzer.visualize_3d_interactive()
#     analyzer.plot_topic_distribution()
    
#     # Optional: Save the model
#     # analyzer.topic_model.save("bertopic_model")
    
#     # Optional: Get topic info as DataFrame
#     topic_info = analyzer.topic_model.get_topic_info()
#     print("\nTopic Information DataFrame:")
#     print(topic_info)
    
#     return analyzer




"""
# Todo: Clear out the following code. 
"""


# Example arguments — set these outside or pass them in
# text_model_path = "Qwen/Qwen2-1.5B"
# max_token_length = 128
# device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch, tokenizer, max_length):
    # Tokenize a batch of texts
    tokenized = tokenizer(
        batch,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


def load_qwen(
    text_inp,
    text_model_path,
    batch_size=8,
    num_workers=2,
    device="cuda",
):
    dataset = TextDataset(text_inp)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=lambda x: collate_fn(x, tokenizer, max_token_length)
                )
   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(text_model_path)
    model = model.to(device)
    
    model.eval()
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    all_embeddings = []
    with torch.no_grad():

        # total_pca_time = 0
        for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(dataloader)):
            if batch_idx == 1: 
                start = time()
                
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
    
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            last_hidden_state = outputs.last_hidden_state  # convert fp16 → fp32 for stability
            all_embeddings.append(last_hidden_state)
            torch.cuda.empty_cache()

        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, seq_len, hidden_dim)
        # start_pca = time()
        # all_embeddings = pca_lowrank(all_embeddings.reshape(len(text_inp), -1), top_n_components=100) 
        # end_pca =  time()
        # total_pca_time += start_pca - end_pca
    end = time()
    print(f"total time taken is: {end - start}")
    # print(f"Total time taken for PCA is: {total_pca_time}")
    return all_embeddings
