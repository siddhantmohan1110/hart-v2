"""
Example of using HART with cluster-based context switching.

This demonstrates how to:
1. Cluster prompts to find centroids
2. Use cluster centroid context for first 4 pyramid levels
3. Use original prompt context for remaining levels
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel

def cluster_prompts_get_centroid(
    prompts: list,
    text_model,
    text_tokenizer,
    num_clusters: int = 5
):
    """
    Cluster prompts and return cluster centroids.
    
    Args:
        prompts: List of prompt strings
        text_model: Pre-trained language model
        text_tokenizer: Text tokenizer
        num_clusters: Number of clusters to create
    
    Returns:
        centroids: List of centroid context tensors
        assignments: Cluster assignment for each prompt
    """
    # Encode all prompts
    encoded_prompts = []
    for prompt in prompts:
        with torch.no_grad():
            tokens = text_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
            with torch.cuda.amp.autocast():
                outputs = text_model(tokens.cuda())
                # Get the pooled output (adjust based on your model)
                context = outputs.last_hidden_state[:, 0, :].cpu()  # [1, context_dim]
        encoded_prompts.append(context)
    
    # Stack into numpy array for clustering
    prompt_features = torch.cat(encoded_prompts, dim=0).numpy()  # [num_prompts, context_dim]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    assignments = kmeans.fit_predict(prompt_features)
    centroids = kmeans.cluster_centers_
    
    return centroids, assignments


def generate_with_cluster_support(
    prompt: str,
    prompt_context_tensor: torch.Tensor,  # Original prompt's context
    cluster_centroid_tensor: torch.Tensor,  # Cluster centroid's context
    model,  # HART model
    text_model,
    text_tokenizer,
    context_mask,
    context_position_ids,
    cfg: float = 4.5,
    seed: int = 0,
    use_cluster_levels: int = 4,
):
    """
    Generate an image using cluster-based context switching.
    
    For first 4 pyramid levels: uses cluster centroid context
    From 5th level onwards: uses original prompt context + previously generated tokens
    
    Args:
        prompt: The original prompt string
        prompt_context_tensor: Context tensor from original prompt (shape: [1, context_token, context_dim])
        cluster_centroid_tensor: Context tensor from cluster centroid (shape: [1, context_token, context_dim])
        model: HART model (already loaded)
        text_model: Language model
        text_tokenizer: Text tokenizer
        context_mask: Context mask tensor
        context_position_ids: Position IDs for context
        cfg: Classifier-free guidance strength
        seed: Random seed
        use_cluster_levels: Number of levels to use cluster centroid (default: 4)
    
    Returns:
        Generated image tensor [1, 3, 1024, 1024]
    """
    
    # Call the modified autoregressive_infer_cfg
    with torch.no_grad():
        output_img = model.autoregressive_infer_cfg(
            B=1,  # Batch size
            label_B=prompt_context_tensor,
            cfg=cfg,
            g_seed=seed,
            more_smooth=False,
            context_position_ids=context_position_ids,
            context_mask=context_mask,
            cluster_centroid_context=cluster_centroid_tensor,  # NEW PARAMETER
            use_cluster_levels=use_cluster_levels,  # NEW PARAMETER: first 4 levels
        )
    
    return output_img


def main():
    """
    Example workflow:
    1. Load models
    2. Cluster your training/validation prompts
    3. For each new prompt, find its cluster
    4. Generate image using cluster centroid for early levels
    """
    
    # Example prompts
    prompts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "Modern architecture in a cityscape",
        # ... more prompts
    ]
    
    # Load your models (pseudo-code)
    # model = load_hart_model(...)
    # text_model = load_text_model(...)
    # text_tokenizer = load_tokenizer(...)
    
    # Cluster prompts (once, offline)
    # centroids, assignments = cluster_prompts_get_centroid(
    #     prompts, text_model, text_tokenizer, num_clusters=5
    # )
    
    # For each new prompt:
    new_prompt = "A dog playing in a park"
    
    # 1. Encode new prompt
    # new_context_tensor = encode_prompt(new_prompt, text_model)
    
    # 2. Find which cluster this prompt belongs to (optional: can be nearest neighbor)
    # distances = np.linalg.norm(new_context_tensor - centroids, axis=1)
    # cluster_idx = np.argmin(distances)
    # centroid_context = torch.tensor(centroids[cluster_idx]).unsqueeze(0)
    
    # 3. Generate with cluster support
    # output_img = generate_with_cluster_support(
    #     prompt=new_prompt,
    #     prompt_context_tensor=new_context_tensor,
    #     cluster_centroid_tensor=centroid_context,
    #     model=model,
    #     text_model=text_model,
    #     text_tokenizer=text_tokenizer,
    #     context_mask=context_mask,
    #     context_position_ids=context_position_ids,
    #     cfg=4.5,
    #     seed=42,
    #     use_cluster_levels=4,  # Use centroid for first 4 levels
    # )
    
    print("Example code structure created. Implement the actual model loading and generation logic.")


if __name__ == "__main__":
    main()

