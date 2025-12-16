# HARTv2 : A more efficient HART

by [Siddhant Mohan](mailto:sm12766@nyu.edu), [Jay Daftari](mailto:jd5829@nyu.edu) and [Athul Radhakrishnan](mailto:ar6316@nyu.edu) under the guidance of [Prof. Sai Qian Zhang](mailto:sai.zhang@nyu.edu).

## Abstract

Improvements to make HART even more efficient.

## Setup on NYU HPC

Login to NYU HPC with your netID and follow the [official instructions](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda) to set up a Singularity container with Miniconda, and use ```cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif``` as the Singularity image. Use only a NVIDIA A100 GPU.

Inside the Singularity container, set up Git LFS.
```bash
conda install -c conda-forge -y git-lfs
git lfs install
```

Download the repositories in the main directory and pull the weight files.
```bash
mkdir /scratch/netID/main_dir
cd /scratch/netID/main_dir
git clone https://github.com/siddhantmohan1110/hart-v2
git clone https://huggingface.co/mit-han-lab/Qwen2-VL-1.5B-Instruct
git clone https://huggingface.co/mit-han-lab/hart-0.7b-1024px
git clone https://huggingface.co/google/shieldgemma-2b

cd /scratch/netID/main_dir/hart-0.7b-1024px/llm && git lfs pull
cd /scratch/netID/main_dir/Qwen2-VL-1.5B-Instruct && git lfs pull
cd /scratch/netID/main_dir/shieldgemma-2b && git lfs pull
```

Install dependencies.
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.6.0+cu126 torchvision==0.21.0+cu126
pip install --extra-index-url https://download.pytorch.org/whl/cu126 xformers==0.0.29.post2

cd hart-v2
pip install -e .
cd hart/kernels
python setup.py install
```

Note: ShieldGemma-2B from Google DeepMind is used for filtering out unsafe prompts in the demo.

Launch the Gradio demo.

```bash
cd main_dir
python ./hart-v2/app.py --model_path ./hart-0.7b-1024px/llm --text_model_path ./Qwen2-VL-1.5B-Instruct --shield_model_path ./shieldgemma-2b
```

## Clustering and Sampling Workflows

Two entrypoints drive offline clustering and downstream sampling:

- `clustering_test.py` clusters prompts, summarizes clusters, and (optionally) generates shared f_hat centroids.
- `sample.py` consumes clustered artifacts or baseline prompts to render grids and/or per-prompt images (shared or non-shared HART).

### clustering_test.py (cluster + summarize)

Key arguments (abbreviated):
- `--dataset` (`imagenet`|`mjhq`) and dataset paths (`--imagenet_class_labels_path`, `--mjhq-metadata-path`).
- Embeddings: `--embedding_model` (`clip`|`siglip`|`qwen`) and `--embedding_batch_size`.
- Clustering: `--clustering_algo` (`agglomerative`|`kmeans`|`hdbscan`), `--n_clusters` (when applicable).
- Summaries: `--summarizer_model_path`, `--max_prompts_per_cluster`.
- HART generation: `--model_path`, `--text_model_path`, `--alpha` (centroid stage), `--use_ema`, `--cfg`, `--more_smooth`.
- Output control: `--experiment_name` (folder name), `--generate_centroid_grid` (create centroid grid), `--stop_with_centroid_summaries` (skip HART), and filenames are prefixed with the experiment name.

Typical run:
```bash
python clustering_test.py \
  --experiment_name exp1_clip_hac \
  --dataset imagenet \
  --embedding_model clip \
  --clustering_algo agglomerative --n_clusters 100 \
  --summarizer_model_path ./../saved_models/Qwen2-VL-1.5B-Instruct/ \
  --model_path ./../saved_models/hart-0.7b-1024px/llm \
  --text_model_path ./../saved_models/Qwen2-VL-1.5B-Instruct/ \
  --generate_centroid_grid
```
Outputs land in `exp1_clip_hac/`:
- `{experiment}_cluster_prompts.json`
- `{experiment}_summary_centroids.json`
- `{experiment}_fhat_centroids.pt` (shared centroids)
- `{experiment}_cluster_centroids_grid.png` (optional)
- `{experiment}_timing_profile.json`

### sample.py (render grids and individual images)

Modes are selected via:
- `--clustered_prompts`: use clustered artifacts under `experiment_name/`.
- `--shared_hart`: use shared f_hat centroids (requires clustered mode and `--alpha`).
- Otherwise, baseline prompts come from `--dataset` (Imagenet/MJHQ) or fall back to built-in defaults.

I/O locations (under `experiment_name/`):
- Grids: `cluster_grids_shared/`, `cluster_grids_nonshared/`, `baseline_grids/`
- Individual images: `cluster_images_shared/`, `cluster_images_nonshared/`, `baseline_images/`

Primary arguments:
- Data: `--experiment_name`, `--cluster_prompts_path`, `--fhat_path`, `--dataset`, `--imagenet_class_labels_path`, `--mjhq_metadata_path`.
- Inference: `--alpha`, `--cfg`, `--seed`, `--batch_size`, `--use_ema`, `--use_llm_system_prompt`, `--max_token_length`.
- Outputs: `--generate_grids` (create grids), `--grid_nrow`, `--grid_full_res` (disable downsample), `--save_individual_images`, `--num_images_per_prompt`, `--resize_individual_to`.
- Timing (optional): `--enable_timing`, `--warmup_iterations` (warmup batches before timing).

Examples:
1) Clustered + shared HART grids only:
```bash
python sample.py --experiment_name exp1_clip_hac --clustered_prompts --shared_hart --generate_grids
```
2) Clustered non-shared grids and per-prompt images (e.g., for FID):
```bash
python sample.py --experiment_name exp1_clip_hac --clustered_prompts \
  --generate_grids --save_individual_images --num_images_per_prompt 5 \
  --resize_individual_to 256
```
3) Baseline (no clustering) per-prompt Imagenet set for FID:
```bash
python sample.py --dataset imagenet --generate_grids --save_individual_images \
  --num_images_per_prompt 5 --resize_individual_to 256
```

## Acknowledgements

Our codebase is inspired by amazing open source research projects such as [HART](https://github.com/mit-han-lab/hart) and [VAR](https://github.com/FoundationVision/VAR).
