import sys
sys.modules['tensorflow'] = None
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import json
from time import time
import argparse
import copy
from typing import Any, List 
import numpy as np
import torchvision
from PIL import Image
import re
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
from sklearn.cluster import KMeans
import torch

# from cuml.cluster import HDBSCAN
from hdbscan import HDBSCAN

# from hart.clustering import Topic2VecClustering
from hart.clustering.algos.bert_topic import BERTopicAnalyzer, load_qwen
from hart.utils.datasets import load_mjhq

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I


# def test_TTV(prompts):
#     ttv = Topic2VecClustering()
#     ttv.init_model(prompts)
#     # Simplified version
#     fig_simple = ttv.plot_interactive_with_centroids(
#         file_name="simple_centroids_0_05_dist",
#         save_path="./",
#         show_fig=True
#     )


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
    kmeans = KMeans(n_clusters=100)

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


def summarizer(summary_prompts, model, tokenizer, batch_size=4):
    """
    Generate summaries in fixed-order batches to avoid GPU OOM.
    Uses a DataLoader with shuffle disabled so output order matches input order.
    """
    device = next(model.parameters()).device  # torch.device('cuda:0'), etc.

    if isinstance(summary_prompts, str):
        summary_prompts = [summary_prompts]

    data_loader = DataLoader(
        summary_prompts,
        batch_size=batch_size,
        shuffle=False,
    )

    summaries = []

    # Remove banned meta words/phrases.
    banned_terms = [
        "prompt",
        "prompts",
        "answer",
        "inference",
        "summary",
        "cluster",
        "collection",
        "various",
        "depicting",
        "output",
        "to generate",
        "the image should",
        "diverse",
        "images",
        "visual similarity",
        "objects",
        "subject",
        "scene showing",
        "categories",
    ]

    for batch_prompts in data_loader:
        # Tokenize and generate summary for the current batch
        inputs = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            truncation=True,
            max_length=450,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        # Decode only the newly generated tokens (skip the input prompt)
        input_lengths = inputs["attention_mask"].sum(dim=1)

        for idx, summary_ids in enumerate(outputs):
            start = input_lengths[idx].item()
            generated_ids = summary_ids[start:]
            summary_raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Normalize whitespace, drop newlines, and keep only the first sentence.
            summary = " ".join(summary_raw.split()).replace("\\", "")
            first_sentence = re.split(r"[.\n]", summary, maxsplit=1)[0].strip()
            summary = first_sentence or summary
            # Keep only letters, periods, commas, and spaces.
            summary = re.sub(r"[^A-Za-z., ]+", "", summary)
        
            for term in banned_terms:
                summary = re.sub(rf"\b{re.escape(term)}\b", "", summary, flags=re.IGNORECASE)
            summary = " ".join(summary.split()).strip("., ")

            summaries.append(summary)

    return summaries

def main(
        prompts,
        text_model_path,
        limit=10**5,
        clustering_algo="hdbscan",
        batch_size=128,
        summarizer_batch_size=4,
        **hdb_configs):

    # Load qwen model for summarization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        text_model_path,
        padding_side="left",
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        text_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    qwen_model.config.pad_token_id = qwen_tokenizer.pad_token_id

    if limit: 
        prompts = prompts[:limit]

    visual_summary_prompts = [f"""
    You are an expert visual prompt engineer for the HART (Hybrid Autoregressive Transformer) image generation model. Your goal is to convert short, ambiguous ImageNet class labels into rich, unambiguous, photorealistic visual descriptions.

    CRITICAL INSTRUCTION:
    The input labels come from the ImageNet dataset, which is based on the WordNet hierarchy. You must ALWAYS prioritize the WordNet definition of the object.
    - If the label is "Black Widow", you must describe the spider (Latrodectus), NEVER the Marvel character.
    - If the label is "Crane", you must check the context or provide a specific description of the bird (Gruidae) or the construction machine, but default to the most common ImageNet biological class if unsure, or specify the biological distinctiveness.
    - If the label is "Jaguar", describe the cat (Panthera onca), not the car (unless specified).

    Your output format for every label must be:
    [Subject Description] + [Environment/Context] + [Lighting/Style] 

    GUIDELINES:
    1. Subject: Explicitly describe the visual features (color, texture, shape). Use scientific names if helpful for clarity.
    2. Context: Place the object in its natural habitat or typical setting.
    3. Style: Use high-quality keywords (4k, detailed texture, cinematic lighting) to ensure HART generates a high-fidelity image.
    4. Output: Provide ONLY the final prompt text. Do not output conversational filler like "Here is the prompt." 
    5. Output must be 40 tokens or fewer

    Example Input: "Black Widow"
    Example Output: A close-up macro photograph of a Latrodectus spider, commonly known as a black widow, featuring a shiny black bulbous abdomen with a distinctive red hourglass marking. The spider is resting on a chaotic silk web in a dark, shadowy corner. Natural lighting, high contrast, 8k resolution, photorealistic texture.

    Input: {pr} 
    Output: 
    """ 
    for pr in prompts]

    print(visual_summary_prompts[0])

    summary_time = time()

    rich_prompts = summarizer(visual_summary_prompts, qwen_model, qwen_tokenizer, batch_size=summarizer_batch_size)

    save_dir = './samples/'
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + "rich_prompts.txt", 'w') as f: 
        f.write("\n".join(rich_prompts) + "\n")

    prompts = rich_prompts

    for idx, prompt in enumerate(prompts):
        prompts[idx] =  prompt

    if clustering_algo.lower() == "hdbscan":
        algo = HDBSCAN(**hdb_configs) #min_samples=3, gen_min_span_tree=True, prediction_data=True)
    
    else:
        print(f"Provided algo is: {clustering_algo.lower()}")
        print(f"Starting Kmeans")
        algo = KMeans(n_clusters=100) # For Kmeans. 

    summary_time_end = time()

    print(f"Total time taken to generate summaries = {summary_time_end - summary_time}")
    start = time()


    # embedding_model = pipeline("feature-extraction", model="distilbert-base-cased")
    

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    analyzer = BERTopicAnalyzer(clustering_model=algo, min_topic_size=3, documents = prompts, n_components=3, embedding_model=embedding_model)
        
    read_time = time()
    # fit the clustering 
    
    analyzer.fit_model()

    # assign the documents to the class. 
    # analyzer.documents = prompts

    end = time()
    print(f"Input reading IO time = {read_time - start}")
    print(f"Total Time = {end - start}")


    topic_embeddings = analyzer.topic_model.topic_embeddings_

    # Get topic assignments for each document
    topics = analyzer.topic_model.topics_
    topic_info = analyzer.topic_model.get_topic_info()

    # Group prompts by cluster/topic
    cluster_prompts = {}
    for doc_idx, topic_id in enumerate(topics):
        if topic_id not in cluster_prompts:
            cluster_prompts[topic_id] = []
        cluster_prompts[topic_id].append(prompts[doc_idx])

    print(f"Found {len(cluster_prompts)} clusters")

  
    # Generate summaries for each cluster
    summary_centroids = {}

    input_to_summarizer = []

    for topic_id, topic_prompts in cluster_prompts.items():
        print(f"Processing cluster {topic_id} with {len(topic_prompts)} prompts...")

        # Create a prompt for summarization
        prompts_text = "\n".join([f"- {p}" for p in topic_prompts[:100]])  # Limit to first 100 to avoid token limits
        summarization_prompt = f"""You are generating ONE text-to-image prompt to be used directly by an image generation model. You are given an IMAGENET label.

        Write the prompt as if you want the image to be generated, not described. Do NOT describe a list, do NOT mention prompts, summaries, clusters, or collections.

        The value must be a single sentence image-generation prompt.

        Strictly avoid meta or generic phrasing.

        Banned content (must not appear anywhere): prompt, prompts, answer, inference, summary, cluster, collection, various, depicting, output, to generate, the image should, diverse, images, visual similarity, objects, subject, scene showing, categories

        Task:
        From the prompts below, infer the most plausible shared visual concept and write ONE concise, visually grounded image-generation prompt.
                
        Guidelines:
        - Focus on concrete visual attributes: object type, shape, texture, material, color, typical pose or viewpoint, and a likely environment if applicable.
        - Capture what is common across the prompts; ignore rare, weak, or incoherent outliers.
        - If the prompts span unrelated categories, choose ONE dominant and visually distinctive subject and ignore the rest.
        - Do NOT list or reference individual class names.
        - Avoid abstract, symbolic, or non-visual language.
        - Avoid stylistic adjectives unless clearly implied.

        Hard Constraints:
        - Output exactly ONE sentence.
        - Output must be 30 tokens or fewer.
        - No bullet points, lists, quotes, or line breaks.
        - No explanations or commentary.
        - Produce exactly ONE sentence summary and nothing else.

        Prompts:
        {prompts_text}

        Prompt:"""

        input_to_summarizer.append(summarization_prompt)

    summaries = summarizer(input_to_summarizer, qwen_model, qwen_tokenizer, batch_size=summarizer_batch_size)

    for summary, cp, in zip(summaries, cluster_prompts.items()):
        topic_id, _ = cp
        summary_centroids[topic_id] = summary
        print(f"Cluster {topic_id} summary: {summary[:100]}...")

    # Merge clusters that ended up with identical cleaned summaries.
    summary_key_map = {}
    merged_summary_centroids = {}
    merged_cluster_prompts = {}
    merged_info = {}
    for topic_id, summary in summary_centroids.items():
        key = summary.lower()
        if key in summary_key_map:
            primary_id = summary_key_map[key]
            merged_info.setdefault(primary_id, []).append(topic_id)
            merged_cluster_prompts[primary_id].extend(cluster_prompts.get(topic_id, []))
        else:
            summary_key_map[key] = topic_id
            merged_summary_centroids[topic_id] = summary
            merged_cluster_prompts[topic_id] = list(cluster_prompts.get(topic_id, []))
    if merged_info:
        print("\nMerging clusters with identical summaries:")
        for primary_id, merged_ids in merged_info.items():
            print(f"  Keeping {primary_id}, merging {merged_ids}")
    summary_centroids = merged_summary_centroids
    cluster_prompts = merged_cluster_prompts

    # Save summary centroids ordered by increasing cluster_id.
    def _cluster_sort_key(cid):
        cid_str = str(cid)
        if cid_str.lstrip("-").isdigit():
            return (0, int(cid))
        return (1, cid_str)

    ordered_ids = sorted(summary_centroids.keys(), key=_cluster_sort_key)
    ordered_summary_centroids = {cid: summary_centroids[cid] for cid in ordered_ids}
    ordered_cluster_prompts = {cid: cluster_prompts[cid] for cid in ordered_ids}
    prompt_centroid_map = {x: cid for cid in ordered_ids for x in cluster_prompts[cid]}

    # Save summary centroids to file
    with open("summary_centroids.json", "w") as f:
        json.dump(ordered_summary_centroids, f, indent=2)
    # Save prompts grouped by cluster_id to file
    with open("cluster_prompts.json", "w") as f:
        json.dump(ordered_cluster_prompts, f, indent=2)

    with open("prompt_centroid_map.json", "w") as f:
        json.dump(prompt_centroid_map, f, indent=2)



    print(f"\nSaved summaries for {len(summary_centroids)} clusters to summary_centroids.json")

    if args.stop_with_centroid_summaries:
        return None

    # Clean up qwen model before loading HART
    del qwen_model
    del qwen_tokenizer
    torch.cuda.empty_cache()

    # Load HART model for generating f_hats from summaries
    print("\nLoading HART model...")
    hart_model = AutoModel.from_pretrained(args.model_path)
    hart_model = hart_model.to(device)
    hart_model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(hart_model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    # Load text model for encoding summaries
    hart_text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    hart_text_model = AutoModel.from_pretrained(text_model_path).to(device)
    hart_text_model.eval()

    # Generate f_hats for each summary centroid with alpha=3
    fhat_centroids = {}
    cluster_output_images = {}  # Changed to dict to maintain cluster_id association
    alpha = 3
    fhat_save_path = "./fhat_centroids"
    os.makedirs(fhat_save_path, exist_ok=True)

    print(f"\nGenerating f_hats for {len(summary_centroids)} cluster summaries with alpha={alpha}...")

    # Sort cluster IDs to process in order
    sorted_cluster_ids = sorted(summary_centroids.keys(), key=lambda x: int(x) if str(x).lstrip('-').isdigit() else float('inf'))

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            for topic_id in sorted_cluster_ids:
                summary = summary_centroids[topic_id]
                print(f"Processing summary for cluster {topic_id}...")

                # Encode the summary using the same function from sample.py
                (
                    context_tokens,
                    context_mask,
                    context_position_ids,
                    context_tensor,
                ) = encode_prompts(
                    [summary],  # Pass as list
                    hart_text_model,
                    hart_text_tokenizer,
                    args.max_token_length,
                    llm_system_prompt,
                    args.use_llm_system_prompt,
                )

                # Select inference function based on EMA usage
                infer_func = (
                    ema_model.autoregressive_infer_cfg
                    if args.use_ema
                    else hart_model.autoregressive_infer_cfg
                )

                # Forward pass through HART with save_fhat=True and alpha=3
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    save_fhat=True,
                    save_fhat_path=fhat_save_path,
                    alpha=alpha,
                    is_shared_hart=False,
                )

                # Store the output image with cluster_id for ordered grid creation
                cluster_output_images[topic_id] = output_imgs[0]

                # Load the saved f_hat for this cluster
                fhat_file = os.path.join(fhat_save_path, f'fhat_kv_stage_{alpha}.pt')
                if os.path.exists(fhat_file):
                    fhat_data = torch.load(fhat_file)
                    fhat_centroids[topic_id] = fhat_data['f_hat']
                    print(f"Saved f_hat for cluster {topic_id} with shape {fhat_data['f_hat'].shape}")
                    # Clean up the temporary file
                    os.remove(fhat_file)
                else:
                    print(f"Warning: f_hat file not found for cluster {topic_id}")

    # Create and save grid image of all cluster centroids in sorted order
    if cluster_output_images:
        # Stack images in sorted order by cluster_id
        sorted_images = [cluster_output_images[cid] for cid in sorted_cluster_ids]
        cluster_images_tensor = torch.stack(sorted_images)
        grid = torchvision.utils.make_grid(cluster_images_tensor, nrow=min(8, len(cluster_output_images)))
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()
        grid_np = Image.fromarray(grid_np.astype(np.uint8))

        grid_save_path = "cluster_centroids_grid.png"
        grid_np.save(grid_save_path)
        print(f"\nSaved grid of {len(cluster_output_images)} cluster centroid images to {grid_save_path}")
        print(f"Images ordered by cluster_id: {sorted_cluster_ids}")

    # Save all f_hat centroids
    torch.save(fhat_centroids, "fhat_centroids.pt")
    print(f"\nSaved f_hats for {len(fhat_centroids)} clusters to fhat_centroids.pt")

    # Clean up
    del hart_model
    if args.use_ema:
        del ema_model
    del hart_text_model
    del hart_text_tokenizer
    torch.cuda.empty_cache()

    

    # total_time = time.time() - start_time
    # print(f"Generate {len(prompts)} images take {total_time:2f}s.")
    
    # print("Visualizing")
    # analyzer.visualize_3d_interactive()
    # print("finished visualizing")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjhq-meta-path",
        type=str,
        help="The path to MJHQ meta_data.json.",
        default="./../MJHQ-30K/meta_data.json",    
    )

    parser.add_argument(
        "--clustering_algo",
        type=str,
        help="The clustering algorithm to use. We employ HDBSCAN by default.",
        default="KMeans"
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="./../Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="./../hart-0.7b-1024px/llm",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )

    parser.add_argument("--stop_with_centroid_summaries", action="store_true", help="If set, the program will stop after generating centroid summaries.")
    parser.add_argument(
        "--summarizer_batch_size",
        type=int,
        default=32,
        help="Batch size for the Qwen summarizer DataLoader (keeps ordering, no shuffle).",
    )

    args = parser.parse_args()
    clustering_algo = args.clustering_algo
    # prompts = load_mjhq(args.get('mjhq-meta-path'))
    with open("./../ILSVRC2012_devkit_t12/imagenet_classes.txt") as f:
        imagenet_labels = [x.strip() for x in f.readlines()]
    # with open('./../ILSVRC2012_devkit_t12/imagenet_classid_to_label.json') as f:
        # val_map = json.load(f)

    # imagenet_labels = []

    # for label in val_map.values():
    #     # Split at commas → e.g. "tench, Tinca tinca"
    #     parts = label.split(',')
    #     # Clean spaces and lowercase for consistency
    #     parts = [p.strip().lower() for p in parts]
    #     imagenet_labels.extend(parts)

    # Print results
    print("Total labels:", len(imagenet_labels))
    prompts = imagenet_labels

    text_model_path = args.text_model_path
    hdb_config = dict(min_samples=3, gen_min_span_tree=True, prediction_data=True)
    main(
        prompts,
        text_model_path,
        limit=None,
        clustering_algo=clustering_algo,
        batch_size=128,
        summarizer_batch_size=args.summarizer_batch_size,
        **hdb_config,
    )
    # test_BertTopic(prompts, text_model_path)

    # test_TTV(prompts)
