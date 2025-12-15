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
import numpy as np
import torchvision
from PIL import Image
import re
from hart.utils import encode_prompts
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch

# from cuml.cluster import HDBSCAN
from hdbscan import HDBSCAN

# from hart.clustering import Topic2VecClustering
from hart.clustering.algos.bert_topic import BERTopicAnalyzer
from hart.utils.datasets import load_mjhq

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.utils.constants import (
    summarization_prompt_template,
    enrichment_prompt_template,
    banned_meta_terms,
    llm_system_prompt,
)


def load_siglip_embeddings(texts, model_name="siglip", batch_size=32):
    """Load SigLIP model and generate text embeddings."""
    from transformers import AutoProcessor, AutoModel
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in dataloader:
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.get_text_features(**inputs)
            all_embeddings.append(outputs.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()

    return all_embeddings


def load_clip_embeddings(texts, model_name="openai/clip-vit-large-patch14-336", batch_size=32):
    """Load CLIP ViT-L/14@336px model and generate text embeddings."""
    from transformers import CLIPProcessor, CLIPModel
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in dataloader:
            inputs = processor(text=list(batch_texts), return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.get_text_features(**inputs)
            all_embeddings.append(outputs.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()

    return all_embeddings


def load_qwen_embeddings(texts, model_name="Qwen2-VL-1.5B-Instruct/", batch_size=32):
    """Load Qwen model and generate text embeddings from last hidden state."""
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in dataloader:
            inputs = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Use mean pooling of last hidden state
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return all_embeddings


def summarize_clusters(
    cluster_prompts,
    summarizer_model_path,
    max_prompts_per_cluster=100,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_path, padding_side="left")
    summarizer_model = AutoModelForCausalLM.from_pretrained(
        summarizer_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    summary_centroids = {}
    timings = {
        "clusters_processed": len(cluster_prompts),
        "per_cluster_sec": {},
    }
    total_start = time()

    for topic_id, topic_prompts in cluster_prompts.items():
        cluster_start = time()
        prompts_text = "\n".join([f"- {p}" for p in topic_prompts[:max_prompts_per_cluster]])
        summarization_prompt = summarization_prompt_template.format(prompts_text=prompts_text)

        inputs = summarizer_tokenizer(
            summarization_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = summarizer_model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary_raw = summarizer_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        summary = " ".join(summary_raw.split()).replace("\\", "")
        first_sentence = re.split(r"[.\n]", summary, maxsplit=1)[0].strip()
        summary = first_sentence or summary
        summary = re.sub(r"[^A-Za-z., ]+", "", summary)

        for term in banned_meta_terms:
            summary = re.sub(rf"\b{re.escape(term)}\b", "", summary, flags=re.IGNORECASE)
        summary = " ".join(summary.split()).strip("., ")

        summary_centroids[topic_id] = summary
        timings["per_cluster_sec"][str(topic_id)] = time() - cluster_start

    merge_start = time()
    summary_key_map = {}
    merged_summary_centroids = {}
    merged_cluster_prompts = {}
    for topic_id, summary in summary_centroids.items():
        key = summary.lower()
        if key in summary_key_map:
            primary_id = summary_key_map[key]
            merged_cluster_prompts[primary_id].extend(cluster_prompts.get(topic_id, []))
        else:
            summary_key_map[key] = topic_id
            merged_summary_centroids[topic_id] = summary
            merged_cluster_prompts[topic_id] = list(cluster_prompts.get(topic_id, []))
    summary_centroids = merged_summary_centroids
    cluster_prompts = merged_cluster_prompts
    timings["merge_sec"] = time() - merge_start

    def _cluster_sort_key(cid):
        cid_str = str(cid)
        if cid_str.lstrip("-").isdigit():
            return (0, int(cid))
        return (1, cid_str)

    ordered_ids = sorted(summary_centroids.keys(), key=_cluster_sort_key)
    ordered_summary_centroids = {cid: summary_centroids[cid] for cid in ordered_ids}
    ordered_cluster_prompts = {cid: cluster_prompts[cid] for cid in ordered_ids}
    timings["total_sec"] = time() - total_start

    del summarizer_model
    del summarizer_tokenizer
    torch.cuda.empty_cache()
    return ordered_summary_centroids, ordered_cluster_prompts, timings


def generate_rich_prompts(
    prompts,
    enrichment_model_path,
    batch_size=8,
    max_new_tokens=80,
    temperature=0.2,
    top_p=0.9,
    timings=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(enrichment_model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        enrichment_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    enriched = []
    per_batch = {}
    total_start = time()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_rich_prompts = [enrichment_prompt_template.format(pr=pr) for pr in batch_prompts]
        batch_start = time()
        inputs = tokenizer(
            batch_rich_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            )
        per_batch[str(start // batch_size)] = time() - batch_start
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            generated = output[input_len:]
            summary_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            cleaned = " ".join(summary_raw.split()).strip()
            cleaned = re.sub(r"[^A-Za-z., ]+", "", cleaned)
            for term in banned_meta_terms:
                cleaned = re.sub(rf"\b{re.escape(term)}\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = " ".join(cleaned.split()).strip("., ")
            enriched.append(cleaned)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    if timings is not None:
        timings["enrichment_batches_sec"] = per_batch
        timings["enrichment_total_sec"] = time() - total_start
    return enriched


def main(args):

    overall_start = time()
    timings = {
        "meta": {
            "embedding_model": args.embedding_model,
            "clustering_algo": args.clustering_algo,
            "use_ema": args.use_ema,
            "max_prompts_per_cluster": args.max_prompts_per_cluster if hasattr(args, "max_prompts_per_cluster") else 100,
            "summarizer_model_path": args.summarizer_model_path,
        }
    }
    output_dir = args.experiment_name
    os.makedirs(output_dir, exist_ok=True)
    experiment_prefix = args.experiment_name
    cluster_prompts_path = os.path.join(output_dir, f"{experiment_prefix}_cluster_prompts.json")
    summary_centroids_path = os.path.join(output_dir, f"{experiment_prefix}_summary_centroids.json")
    fhat_centroids_path = os.path.join(output_dir, f"{experiment_prefix}_fhat_centroids.pt")
    cluster_grid_path = os.path.join(output_dir, f"{experiment_prefix}_cluster_centroids_grid.png")
    timing_output_path = os.path.join(output_dir, f"{experiment_prefix}_timing_profile.json")

    enrichment_model_path = args.enrichment_model_path
    embedding_model=args.embedding_model
    summarizer_model_path=args.summarizer_model_path
    clustering_algo=args.clustering_algo
    text_model_path=args.text_model_path

    if args.dataset.lower() == "mjhq":
        prompts = load_mjhq(args.mjhq_metadata_path)
    elif args.dataset.lower() == "imagenet":
        with open(args.imagenet_class_labels_path) as f:
            prompts = [x.strip() for x in f.readlines()]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from: imagenet, mjhq")

    timings["meta"]["prompt_count"] = len(prompts)

    if getattr(args, "enrich_prompts", False):
        enrich_start = time()
        prompts = generate_rich_prompts(
            prompts,
            enrichment_model_path,
            batch_size=getattr(args, "enrichment_batch_size"),
            timings=timings,
        )
        timings["enrichment_sec"] = time() - enrich_start

    # Select embedding model for clustering
    embedding_start = time()
    if embedding_model.lower() == "siglip":
        embeddings = load_siglip_embeddings(prompts, model_name="siglip", batch_size=getattr(args, "embedding_batch_size"))
    elif embedding_model.lower() == "clip":
        embeddings = load_clip_embeddings(prompts, model_name="openai/clip-vit-large-patch14-336", batch_size=getattr(args, "embedding_batch_size"))
    elif embedding_model.lower() == "qwen":
        embeddings = load_qwen_embeddings(prompts, model_name=text_model_path, batch_size=getattr(args, "embedding_batch_size"))
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}. Choose from: siglip, clip, qwen")
    timings["embedding_generation_sec"] = time() - embedding_start

    np_embed = embeddings.cpu().numpy()
    del embeddings
    torch.cuda.empty_cache()

    #base_prompt = "You are given a label from ImageNet Classification Dataset. Some labels like Black widow might be ambiguous. Infer to the right meaning from ImageNet class label and generate the image prompt describing the correct visual attributes of the label.\n Label:" 
    for idx, prompt in enumerate(prompts):
        prompts[idx] =  prompt

    clustering_init_start = time()
    if clustering_algo.lower() == "hdbscan":
        hdb_config = dict(min_samples=3, gen_min_span_tree=True, prediction_data=True)
        algo = HDBSCAN(**hdb_config) #min_samples=3, gen_min_span_tree=True, prediction_data=True)
    elif clustering_algo.lower() == "kmeans":
        algo = KMeans(n_clusters=args.n_clusters)
    elif clustering_algo.lower() == "agglomerative":
        algo = AgglomerativeClustering(n_clusters=args.n_clusters, linkage='ward')
    else:
        raise ValueError(f"Unknown clustering algorithm: {clustering_algo}. Choose from: hdbscan, kmeans, agglomerative") 

    start = time()
    analyzer = BERTopicAnalyzer(clustering_model=algo, min_topic_size=3, n_components=3)
    timings["clustering_init_sec"] = time() - clustering_init_start
    read_time = time()
    # fit the clustering 
    analyzer.fit_model(prompts, np_embed)

    # assign the documents to the class. 
    analyzer.documents = prompts

    end = time()
    timings["clustering_fit_sec"] = end - read_time
    timings["clustering_total_sec"] = end - clustering_init_start

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
    timings["cluster_count"] = len(cluster_prompts)

    # Save cluster prompts to file
    save_clusters_start = time()
    cluster_prompts_serializable = {str(k): v for k, v in cluster_prompts.items()}
    with open(cluster_prompts_path, "w") as f:
        json.dump(cluster_prompts_serializable, f, indent=2)
    timings["cluster_prompts_save_sec"] = time() - save_clusters_start

    # Save summary centroids to file
    summary_start = time()
    ordered_summary_centroids, ordered_cluster_prompts, summary_timings = summarize_clusters(
        cluster_prompts,
        summarizer_model_path,
        max_prompts_per_cluster=getattr(args, "max_prompts_per_cluster", 100),
    )
    timings["summarization"] = summary_timings
    timings["summarization"]["total_with_overhead_sec"] = time() - summary_start

    summary_save_start = time()
    with open(summary_centroids_path, "w") as f:
        json.dump(ordered_summary_centroids, f, indent=2)
    with open(cluster_prompts_path, "w") as f:
        json.dump({str(k): v for k, v in ordered_cluster_prompts.items()}, f, indent=2)
    timings["summary_save_sec"] = time() - summary_save_start

    if args.stop_with_centroid_summaries:
        timings["overall_sec"] = time() - overall_start
        if timing_output_path:
            with open(timing_output_path, "w") as f:
                json.dump(timings, f, indent=2)
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load HART model for generating f_hats from summaries
    hart_load_start = time()
    hart_model = AutoModel.from_pretrained(args.model_path)
    hart_model = hart_model.to(device)
    hart_model.eval()
    timings["hart_model_load_sec"] = time() - hart_load_start

    if args.use_ema:
        ema_start = time()
        ema_model = copy.deepcopy(hart_model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )
        timings["ema_load_sec"] = time() - ema_start

    # Load Qwen text model for encoding summaries for HART forward pass
    hart_text_load_start = time()
    hart_text_tokenizer = AutoTokenizer.from_pretrained(text_model_path, padding_side="left")
    hart_text_model = AutoModel.from_pretrained(text_model_path).to(device)
    hart_text_model.eval()
    timings["hart_text_model_load_sec"] = time() - hart_text_load_start

    # Generate f_hats for each summary centroid with alpha=3
    fhat_centroids = {}
    cluster_output_images = {}  # Changed to dict to maintain cluster_id association
    alpha = args.alpha
    stop_after_fhat = not getattr(args, "generate_centroid_grid", False)

    fhat_generation_start = time()
    fhat_cluster_timings = {}

    # Sort cluster IDs to process in order
    sorted_cluster_ids = sorted(ordered_summary_centroids.keys(), key=lambda x: int(x) if str(x).lstrip('-').isdigit() else float('inf'))

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            for topic_id in sorted_cluster_ids:
                cluster_gen_start = time()
                summary = ordered_summary_centroids[topic_id]

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

                # Forward pass through HART to get f_hat
                output_imgs, f_hat = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    alpha=alpha,
                    is_shared_hart=False,
                    return_fhat=True,
                    stop_after_fhat=stop_after_fhat,
                )

                # Store the output image with cluster_id for ordered grid creation
                if output_imgs is not None:
                    cluster_output_images[topic_id] = output_imgs[0]

                # Capture the returned f_hat directly instead of reloading from disk
                fhat_centroids[topic_id] = f_hat.detach().cpu()
                fhat_cluster_timings[str(topic_id)] = time() - cluster_gen_start

    timings["fhat_generation"] = {
        "total_sec": time() - fhat_generation_start,
        "per_cluster_sec": fhat_cluster_timings,
    }

    # Create and save grid image of all cluster centroids in sorted order
    grid_time_start = time()
    if args.generate_centroid_grid and cluster_output_images:
        # Stack images in sorted order by cluster_id
        sorted_images = [cluster_output_images[cid] for cid in sorted_cluster_ids]
        cluster_images_tensor = torch.stack(sorted_images)
        grid = torchvision.utils.make_grid(cluster_images_tensor, nrow=min(8, len(cluster_output_images)))
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()
        grid_np = Image.fromarray(grid_np.astype(np.uint8))

        grid_np.save(cluster_grid_path)
        timings["cluster_grid"] = {
            "save_path": cluster_grid_path,
            "cluster_id_order": sorted_cluster_ids,
            "save_sec": time() - grid_time_start,
        }
    elif args.generate_centroid_grid:
        timings["cluster_grid"] = {"save_sec": time() - grid_time_start, "cluster_id_order": []}
    else:
        timings["cluster_grid"] = {"skipped": True, "save_sec": time() - grid_time_start, "cluster_id_order": []}

    # Save all f_hat centroids
    fhat_save_start = time()
    torch.save(fhat_centroids, fhat_centroids_path)
    timings["fhat_save_sec"] = time() - fhat_save_start
    timings["fhat_saved_count"] = len(fhat_centroids)

    # Clean up
    del hart_model
    if args.use_ema:
        del ema_model
    del hart_text_model
    del hart_text_tokenizer
    torch.cuda.empty_cache()

    timings["overall_sec"] = time() - overall_start
    timings["saved_paths"] = {
        "cluster_prompts": cluster_prompts_path,
        "summary_centroids": summary_centroids_path,
        "fhat_centroids": fhat_centroids_path,
        "cluster_grid": timings.get("cluster_grid", {}).get("save_path"),
    }
    if timing_output_path:
        with open(timing_output_path, "w") as f:
            json.dump(timings, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ***********************************************************
    # Dataset
    # ***********************************************************
    parser.add_argument(
        "--imagenet_class_labels_path",
        type=str,
        help="Path to ImageNet class labels",
        default="./../data/ImageNet/imagenet_classes.txt",
    )
    parser.add_argument(
        "--mjhq-metadata-path",
        type=str,
        help="The path to MJHQ meta_data.json.",
        default="./../data/MJHQ-30K/meta_data.json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Prompt dataset to use: imagenet or mjhq",
        default="imagenet",
    )

    # ***********************************************************
    # Prompt enrichment
    # ***********************************************************
    parser.add_argument(
        "--enrich_prompts",
        action="store_true",
        help="Enrich ImageNet labels into richer visual descriptions before clustering.",
    )
    parser.add_argument(
        "--enrichment_model_path",
        type=str,
        help="Model path to use for rich prompt generation.",
        default="./../saved_models/Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument(
        "--enrichment_batch_size",
        type=int,
        help="Batch size for prompt enrichment.",
        default=8,
    )

    # ***********************************************************
    # Pre-clustering prompt embedding
    # ***********************************************************
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="The embedding model for clustering: siglip, clip (ViT-L/14@336px), or qwen.",
        default="clip",
        choices=["siglip", "clip", "qwen"],
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        help="Batch size for embedding generation.",
        default=128,
    )

    # ***********************************************************
    # Clustering of prompts
    # ***********************************************************
    parser.add_argument(
        "--clustering_algo",
        type=str,
        help="The clustering algorithm to use: hdbscan, kmeans, or agglomerative. We employ HDBSCAN by default.",
        default="agglomerative",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        help="Number of clusters for KMeans or Agglomerative Clustering (not used for HDBSCAN).",
        default=100,
    )

    # ***********************************************************
    # Summarization of clusters
    # ***********************************************************
    parser.add_argument(
        "--summarizer_model_path",
        type=str,
        help="Model path to use for summarization.",
        default="./../saved_models/Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument(
        "--max_prompts_per_cluster",
        type=int,
        help="Maximum prompts per cluster to include in the summarization prompt.",
        default=100,
    )
    
    # ***********************************************************
    # HART
    # ***********************************************************
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="Model path to use for HART text embeddings, HART employs Qwen2-VL-1.5B-Instruct by default.",
        default="./../saved_models/Qwen2-VL-1.5B-Instruct/",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="./../saved_models/hart-0.7b-1024px/llm",
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
    parser.add_argument(
        "--alpha",
        type=int,
        help="Stage index up to which shared HART is used. For example, alpha=3 means stages 0,1,2,3 use shared HART while stages 4+ use prompt-specific HART.",
        default=3,
    )

    # ***********************************************************
    # Output / bookkeeping
    # ***********************************************************
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Directory name where outputs (jsons/images) for this run will be stored.",
        default="exp1",
    )
    parser.add_argument(
        "--generate_centroid_grid",
        action="store_true",
        help="Generate and save grid image of centroid outputs. When enabled, full decoding runs with stop_after_fhat=False.",
    )
    parser.add_argument(
        "--stop_with_centroid_summaries",
        action="store_true",
        help="If set, the program will stop after generating centroid summaries.",
    )

    args = parser.parse_args()

    main(args)
