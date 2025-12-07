import argparse
import json
from time import time

from sklearn.cluster import KMeans
import torch

# from cuml.cluster import HDBSCAN
from hdbscan import HDBSCAN

from hart.clustering import Topic2VecClustering
from hart.clustering.algos.bert_topic import BERTopicAnalyzer, load_qwen
from hart.utils.datasets import load_mjhq



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

    base_prompt = "You are given a label from ImageNet Classification Dataset. Some labels like Black widow might be ambiguous. Infer to the right meaning from ImageNet class label and generate the image prompt describing the correct visual attributes of the label.\n Label:" 

    for idx, prompt in enumerate(prompts):
        prompts[idx] = base_prompt + " " + prompt

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


def main(
        prompts,
        text_model_path,
        limit=10**5,
        clustering_algo="hdbscan",
        batch_size=128,
        **hdb_configs):

    if limit: 
        prompts = prompts[:limit]

    embeddings = load_qwen(prompts, text_model_path, batch_size=batch_size)
    np_embed = embeddings.cpu().numpy()
    del embeddings
    torch.cuda.empty_cache()


    if limit: 
        prompts = prompts[:limit]

    base_prompt = "You are given a label from ImageNet Classification Dataset. Some labels like Black widow might be ambiguous. Infer to the right meaning from ImageNet class label and generate the image prompt describing the correct visual attributes of the label.\n Label:" 
    for idx, prompt in enumerate(prompts):
        prompts[idx] = base_prompt + " " + prompt

    if clustering_algo.lower() == "hdbscan":
        algo = HDBSCAN(**hdb_configs) #min_samples=3, gen_min_span_tree=True, prediction_data=True)
    
    else:
        print(f"Provided algo is: {clustering_algo.lower()}")
        print(f"Starting Kmeans")
        algo = KMeans(n_clusters=100) # For Kmeans. 

    start = time()
    analyzer = BERTopicAnalyzer(clustering_model=algo, min_topic_size=3, n_components=3)
        
    read_time = time()
    # fit the clustering 
    analyzer.fit_model(prompts, np_embed)

    # assign the documents to the class. 
    analyzer.documents = prompts

    


    end = time()
    print(f"Total Time = {end - start}")
    print(f"Input reading IO time = {read_time - start}")


    topic_embeddings = analyzer.topic_model.topic_embeddings_

    
    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for idx, embedding in enumerate(topic_embeddings):
                print(f"Topic {idx}: {embedding}")

                if idx == 5: 
                    start_time = time.time()
                    print(f"Starting time for topic {idx}: {start_time}")

                context_tensor = embedding.to(device).float()
                context_position_ids = torch.zeros(context_tensor.size(0), dtype=torch.long)
                context_mask = torch.ones(context_tensor.size(0), dtype=torch.long)

                infer_func = (
                    ema_model.autoregressive_infer_cfg
                    if args.use_ema
                    else model.autoregressive_infer_cfg
                )
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    save_fhat=False,
                    is_shared_hart=False,
                )

    total_time = time.time() - start_time
    print(f"Generate {len(prompts)} images take {total_time:2f}s.")
    
    # print("Visualizing")
    # analyzer.visualize_3d_interactive()
    # print("finished visualizing")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjhq-meta-path",
        type=str,
        help="The path to MJHQ meta_data.json.",
        default="./MJHQ-30K/meta_data.json",    
    )

    parser.add_argument(
        "--clustering_algo",
        type=str,
        help="The clustering algorithm to use. We employ HDBSCAN by default.",
        default="hdbscan"
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="./../Qwen2-VL-1.5B-Instruct/",
    )


    args = parser.parse_args()
    clustering_algo = args.clustering_algo
    # prompts = load_mjhq(args.get('mjhq-meta-path'))
    with open("data/imagenet_classes.txt") as f:
        imagenet_labels = [x.strip() for x in f.readlines()]
    # with open('./../ILSVRC2012_devkit_t12/imagenet_classid_to_label.json') as f:
    #     val_map = json.load(f)

    # imagenet_labels = []

    # for label in val_map.values():
    #     # Split at commas → e.g. "tench, Tinca tinca"
    #     parts = label.split(',')
    #     # Clean spaces and lowercase for consistency
    #     parts = [p.strip().lower() for p in parts]
    #     imagenet_labels.extend(parts)

    # Print results
    print("Total labels:", len(imagenet_labels))
    print(imagenet_labels[:10])

    prompts = imagenet_labels

    text_model_path = args.text_model_path
    hdb_config = dict(min_samples=3, gen_min_span_tree=True, prediction_data=True)
    main(prompts, text_model_path, limit = None, clustering_algo=clustering_algo, batch_size=128, **hdb_config)
    # test_BertTopic(prompts, text_model_path)

    # test_TTV(prompts)