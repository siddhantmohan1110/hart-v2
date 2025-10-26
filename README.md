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
pip install top2vec[sentence_transformers] # for clustering
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


**Install BertTopic for clustering**

```
pip install bertopic


# Install cuML for UMAP and HDBSCAN GPU implementation.
pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
pip install --upgrade cupy-cuda12x -f https://pip.cupy.dev/aarch64
```

## Acknowledgements

Our codebase is inspired by amazing open source research projects such as [HART](https://github.com/mit-han-lab/hart) and [VAR](https://github.com/FoundationVision/VAR).
