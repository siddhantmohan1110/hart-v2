import argparse
from hart.clustering import Topic2VecClustering
from hart.utils.datasets import load_mjhq
from time import time
from hart.clustering.algos.bert_topic import BERTopicAnalyzer, load_qwen

from cuml.cluster import HDBSCAN
from sklearn.cluster import KMeans
import torch 



def test_TTV(prompts):
    ttv = Topic2VecClustering()
    ttv.init_model(prompts)
    # Simplified version
    fig_simple = ttv.plot_interactive_with_centroids(
        file_name="simple_centroids_0_05_dist",
        save_path="./",
        show_fig=True
    )


def test_BertTopic(prompts, text_model_path, limit=10**5):

    if limit: 
        prompts = prompts[:limit]

    embeddings = load_qwen(prompts, text_model_path, batch_size=128)
    np_embed = embeddings.cpu().numpy()
    del embeddings
    torch.cuda.empty_cache()


    if limit: 
        prompts = prompts[:limit]

    hdbscan = HDBSCAN(min_samples=3, gen_min_span_tree=True, prediction_data=True)
    kmeans = KMeans(n_clusters=50)

    for cls in [hdbscan, kmeans]: 
        start = time()
        analyzer = BERTopicAnalyzer(clustering_model=cls, min_topic_size=3, n_components=3)
            
            # Choose data source (comment/uncomment as needed)
        # Option 1: Use custom documents
        # analyzer.create_custom_documents()
        read_time = time()
        analyzer.fit_model(prompts, np_embed)
        analyzer.documents = prompts

        # Option 2: Load from 20 newsgroups (uncomment to use)
        # analyzer.load_sample_data(n_samples=500)
        
        # Option 3: Use your own documents (uncomment and modify)
        # analyzer.load_data()
        # analyzer.documents = your_documents
        
        # Fit the model
        # analyzer.fit_model()
        end = time()
        print(f"Total Time = {end - start}")
        print(f"Input reading IO time = {read_time - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjhq-meta-path",
        type=str,
        help="The path to MJHQ meta_data.json.",
        default="./MJHQ-30K/meta_data.json",    
    )

    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )


    args = parser.parse_args()
    prompts = load_mjhq(args.get('mjhq-meta-path'))
    text_model_path = args.get('text_model_path')
    test_BertTopic(prompts, text_model_path)

    # test_TTV(prompts)