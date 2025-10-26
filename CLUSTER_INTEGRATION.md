# Cluster-Based Context Switching for HART

## Overview

This modification enables HART to use cluster centroid contexts for the early pyramid levels and original prompt contexts for later levels, implementing a hybrid generation strategy.

## Changes Made

### 1. Modified `autoregressive_infer_cfg` Function

**Location**: `hart/modules/models/transformer/hart_transformer_t2i.py`

**New Parameters**:
- `cluster_centroid_context` (Optional[torch.Tensor]): Context tensor from cluster centroid
- `use_cluster_levels` (int): Number of pyramid levels to use cluster centroid (default: 4)

### 2. Key Logic Addition

The generation now switches contexts at level 5:
- **Levels 0-4** (si < 4): Use `cluster_centroid_context`
- **Levels 5+** (si >= 4): Use `original_sos` (original prompt context)

This allows:
- Coarse structure generation guided by cluster commonalities
- Fine detail generation guided by the specific prompt

## Usage Example

```python
import torch

# 1. Assume you have a cluster centroid context
cluster_centroid = torch.randn(1, 77, 768)  # Example: [B, context_token, context_dim]
original_prompt = torch.randn(1, 77, 768)

# 2. Prepare other necessary tensors
context_mask = torch.ones(1, 77).long().cuda()
context_position_ids = torch.arange(77).unsqueeze(0).cuda()

# 3. Call the generation function
output_image = model.autoregressive_infer_cfg(
    B=1,
    label_B=original_prompt,
    cluster_centroid_context=cluster_centroid,  # NEW
    use_cluster_levels=4,  # NEW: use cluster for first 4 levels
    cfg=4.5,
    g_seed=42,
    context_position_ids=context_position_ids,
    context_mask=context_mask,
)
```

## Implementation Details

### Context Switching Logic

```python
# In the autoregressive loop (around line 400)
if si < use_cluster_levels and cluster_centroid_sos is not None:
    # Use cluster centroid context for early levels
    cond_BD = cluster_centroid_sos
else:
    # Switch to original context for later levels
    cond_BD = original_sos
```

### How It Works

1. **Initialization**: Both original and cluster centroid contexts are prepared with CFG duplication
2. **Early Levels (0-3)**: Generate tokens using cluster centroid's context → common coarse structure
3. **Later Levels (4+)**: Generate tokens using original prompt's context → specific fine details
4. **Result**: Image combines cluster-common structure with prompt-specific details

## Benefits

1. **Structural Consistency**: Early levels benefit from cluster averages
2. **Diversity**: Later levels maintain prompt specificity
3. **Efficiency**: Can pre-compute cluster centroids offline
4. **Quality**: May improve generation quality for prompts within similar clusters

## Clustering Recommendation

You can use any clustering algorithm:

```python
from sklearn.cluster import KMeans
from bertopic import BERTopic  # or any other method

# Option 1: K-means on encoded contexts
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_transform(encoded_contexts)

# Option 2: Topic modeling
topic_model = BERTopic()
topics = topic_model.fit_transform(documents)
```

## Integration with Your Codebase

To integrate this with your clustering setup:

1. **Cluster your training prompts** to find centroids
2. **For each new prompt**:
   - Find its cluster
   - Get cluster centroid's context tensor
   - Generate with both contexts

See `example_clustered_generation.py` for a complete example.

## Testing

The modification is backward-compatible. If you don't pass `cluster_centroid_context`, it behaves exactly like the original HART.

