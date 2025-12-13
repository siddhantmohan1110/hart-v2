import argparse
import json
import time
from tqdm import tqdm

from sklearn.cluster import KMeans
import torch
import os

# from cuml.cluster import HDBSCAN
from hdbscan import HDBSCAN
from transformers import AutoModel, AutoTokenizer

from hart.clustering import Topic2VecClustering
from hart.clustering.algos.bert_topic import BERTopicAnalyzer, load_qwen
from hart.utils.datasets import load_mjhq
from sample import save_images



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
        start = time.time()
        analyzer = BERTopicAnalyzer(clustering_model=cls, min_topic_size=3, n_components=3)
            
            # Choose data source (comment/uncomment as needed)
        # Option 1: Use custom documents
        # analyzer.create_custom_documents()
        read_time = time.time()
        analyzer.fit_model(prompts, np_embed)

        # Option 2: Load from 20 newsgroups (uncomment to use)
        # analyzer.load_sample_data(n_samples=500)
        
        # Option 3: Use your own documents (uncomment and modify)
        # analyzer.load_data()
        # analyzer.documents = your_documents
        
        # Fit the model
        # analyzer.fit_model()
        end = time.time()
        print(f"Total Time = {end - start}")
        print(f"Input reading IO time = {read_time - start}")


# def clear_kv_cache(model):
#     # Ensure we start timing from a clean cache state
#     for module in model.modules():
#         if hasattr(module, "kv_caching"):
#             module.kv_caching(False)


def build_context_from_topics(topic_embeddings, context_dim, context_token, device):
    embedding_tensor = torch.as_tensor(topic_embeddings, device=device, dtype=torch.float32)
    if embedding_tensor.dim() == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)

    if embedding_tensor.shape[1] % context_dim != 0:
        raise ValueError(
            f"Topic embedding length {embedding_tensor.shape[1]} not divisible by context_dim {context_dim}."
        )

    tokens_per_context = embedding_tensor.shape[1] // context_dim
    context_tensor = embedding_tensor.view(embedding_tensor.shape[0], tokens_per_context, context_dim)

    # Adjust to the model's expected number of context tokens
    if tokens_per_context > context_token:
        context_tensor = context_tensor[:, :context_token, :]
    elif tokens_per_context < context_token:
        pad_len = context_token - tokens_per_context
        pad = torch.zeros(
            (context_tensor.shape[0], pad_len, context_dim),
            device=device,
            dtype=context_tensor.dtype,
        )
        context_tensor = torch.cat((context_tensor, pad), dim=1)

    context_mask = torch.ones(
        (context_tensor.size(0), context_tensor.size(1)),
        dtype=torch.long,
        device=device,
    )
    context_position_ids = torch.arange(context_tensor.size(1), device=device).unsqueeze(0).expand_as(context_mask)
    return context_tensor, context_mask, context_position_ids


def main(
        prompts,
        text_model_path,
        limit=10**5,
        clustering_algo="hdbscan",
        batch_size=128,
        infer_batch_size=8,
        decode_context=False,
        decode_batch_size=32,
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

    start = time.time()
    analyzer = BERTopicAnalyzer(clustering_model=algo, min_topic_size=3, n_components=3)
        
    read_time = time.time()
    # fit the clustering 
    analyzer.fit_model(prompts, np_embed)

    # assign the documents to the class. 
    analyzer.documents = prompts


    end = time.time()

    
    print(f"Total Time = {end - start}")
    print(f"Input reading IO time = {read_time - start}")
    print(f"Total Time = {end - start}")


    topic_embeddings = analyzer.topic_model.topic_embeddings_
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = AutoModel.from_pretrained(args.hart_model_path)
    model = model.to(device)
    model.eval()
    context_dim = getattr(model, "context_dim", None)
    context_token = getattr(model, "context_token", None)
    print(f"Embeddings = {len(topic_embeddings)}")
    if context_dim is None or context_token is None:
        raise ValueError("Loaded model is missing expected context dimensions.")

    # if args.use_ema:
    #     ema_model = copy.deepcopy(model)
    #     ema_model.load_state_dict(
    #         torch.load(os.path.join(args.model_path, "ema_model.bin"))
    #     )

    context_tensor, context_mask, context_position_ids = build_context_from_topics(
        topic_embeddings, context_dim, context_token, device
    )
    print(f"Prepared {context_tensor.size(0)} context(s) with shape {context_tensor.shape}.")

    infer_batch_size = max(1, min(infer_batch_size, context_tensor.size(0)))
    warmup_batch = min(infer_batch_size, context_tensor.size(0))

    infer_func = model.autoregressive_infer_cfg

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16, cache_enabled=True):

            # Warm-up on a small batch to avoid cold-start timing and OOMs
            warmup_context = context_tensor[:warmup_batch]
            warmup_mask = context_mask[:warmup_batch]
            warmup_pos = context_position_ids[:warmup_batch]
            for _ in tqdm(range(args.warmup_iter)):
                output_imgs = infer_func(
                    B=warmup_context.size(0),
                    label_B=warmup_context,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=warmup_pos,
                    context_mask=warmup_mask,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.time()
            output_imgs = []
            for start_idx in tqdm(range(0, context_tensor.size(0), infer_batch_size)):
                end_idx = min(start_idx + infer_batch_size, context_tensor.size(0))
                batch_ctx = context_tensor[start_idx:end_idx]
                batch_mask = context_mask[start_idx:end_idx]
                batch_pos = context_position_ids[start_idx:end_idx]
                batch_imgs = infer_func(
                    B=batch_ctx.size(0),
                    label_B=batch_ctx,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=batch_pos,
                    context_mask=batch_mask,
                    save_fhat=True,
                    is_shared_hart=False,
                )
                output_imgs.append(batch_imgs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            print(f"Total lengt of output images are: {len(output_imgs)}")
            output_imgs = torch.cat(output_imgs, dim=0)

    total_time = time.time() - start_time
    per_image = total_time / max(1, context_tensor.size(0))
    print(f"Generate {context_tensor.size(0)} topic images take {total_time:2f}s ({per_image:2f}s/image).")
    print(f"Total images = {len(output_imgs)}")

    prompt_subset = prompts[: context_tensor.size(0)]
    save_images(
        output_imgs.clone(), args.sample_folder_dir, args.store_seperately, prompt_subset
    )

    if decode_context:
        # Context tensor contains embeddings; only decode when tensor already stores token ids.
        if context_tensor.dtype not in (torch.int32, torch.int64):
            print("Skipping context decode: context_tensor is float embeddings, not token ids.")
        else:
            print("here")
            tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            decode_out_path = os.path.join(args.sample_folder_dir, "decoded_contexts.txt")
            os.makedirs(args.sample_folder_dir, exist_ok=True)
            decoded_texts = []
            for start_idx in range(0, context_tensor.size(0), max(1, decode_batch_size)):
                end_idx = min(start_idx + decode_batch_size, context_tensor.size(0))
                batch_ids = context_tensor[start_idx:end_idx].cpu().tolist()
                decoded_texts.extend(
                    tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
                )
            with open(decode_out_path, "w") as f:
                f.write("\n".join(decoded_texts))
            print(f"Decoded {len(decoded_texts)} context entries saved to {decode_out_path}")
    
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
        default="Kmeans"
    )

    parser.add_argument(
        "--hart_model_path",
        type=str,
        help="The path to HART model.",
        default="./../hart-0.7b-1024px/llm/",
    )

    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )

    parser.add_argument("--warmup_iter", type=int, default = 5)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--infer_batch_size",
        type=int,
        help="Batch size for image generation; reduces GPU memory use.",
        default=8,
    )
    parser.add_argument(
        "--decode_context",
        help="Attempt to decode context tensor as token IDs in batches.",
        action="store_true",
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=32,
        help="Batch size for context decoding.",
    )

    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="samples/",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )

    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct/",
    )


    args = parser.parse_args()
    clustering_algo = args.clustering_algo

    # prompts = load_mjhq(args.get('mjhq-meta-path'))
    # with open("data/imagenet_classes.txt") as f:
    #     imagenet_labels = [x.strip() for x in f.readlines()]
    with open('./../ILSVRC2012_devkit_t12/imagenet_classid_to_label.json') as f:
        val_map = json.load(f)

    imagenet_labels = []

    for label in val_map.values():
        # Split at commas → e.g. "tench, Tinca tinca"
        parts = label.split(',')
        # Clean spaces and lowercase for consistency
        parts = [p.strip().lower() for p in parts]
        imagenet_labels.extend(parts)

    # Print results
    print("Total labels:", len(imagenet_labels))
    print(imagenet_labels[:10])

    prompts = imagenet_labels

    text_model_path = args.text_model_path
    hdb_config = dict(min_samples=3, gen_min_span_tree=True, prediction_data=True)
    main(
        prompts,
        text_model_path,
        limit=None,
        clustering_algo=clustering_algo,
        batch_size=128,
        infer_batch_size=args.infer_batch_size,
        decode_context=args.decode_context,
        decode_batch_size=args.decode_batch_size,
        **hdb_config,
    )
    # test_BertTopic(prompts, text_model_path)

    # test_TTV(prompts)
