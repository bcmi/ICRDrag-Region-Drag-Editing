# ICRDrag: Drag Any Region to Any Shape  [![Static Badge](https://img.shields.io/badge/Paper-red)](https://arxiv.org/abs/2606.25907) [![Static Badge](https://img.shields.io/badge/Demo-green)](https://drag.ustcnewly.com/)

This is the official repository for the following paper:

> **In-context Region-based Drag: Drag Any Region to Any Shape** [[arXiv]](https://arxiv.org/abs/2606.25907) <br>
>
> Jiacheng Sui<sup>†</sup>, Tianyu Hao<sup>†</sup>, Bingjie Gao, Li Niu*, Guangtao Zhai <br>
> Accepted by ECCV, 2026. <sup>†</sup> equal contributions. * corresponding author. 

ICRDrag is a drag-style image editing model which supports dragging multiple source regions to their corresponding target regions. Such model can be used for fine-grained geometric editing like pose adjustment and shape transformation. 

![ICRDrag teaser](Figure1.png)

## Online Demo

Try the ICRDrag demo at https://drag.ustcnewly.com/. Users can draw multiple pairs of source and target regions in different colors, then drag each source region to its corresponding target region. 

Note that ICRDrag does not guarantee that every case reaches the ideal edit in a single inference run; some cases may require multiple runs. If undesired changes appear in other areas, you can add anchor-like source-target region pairs to keep the rest of the scene unchanged.

[![]](https://github.com/user-attachments/assets/4c7dd7d0-142d-46b0-b5dd-8f2e16cde707)

## Dataset Overview

We release the PRD training set and two inference/evaluation benchmarks used by
ICRDrag.

PRD is the training dataset for region-based image dragging. It contains
287,153 tuples:

```text
source image, source region mask, target image, target region mask
```

PRDBench is the main quantitative benchmark with 1000 tuples. DragBench is also
provided for evaluating ICRDrag on DragBench-DR and DragBench-SR.

## Dataset Download

Datasets and our results are available from
[[Baidu_Cloud]](https://pan.baidu.com/s/10c47ayBtbluhMzJ-Ocl3LQ?pwd=a83m)
(access code: `a83m`). 

We release the following assets:

```text
PRD_train_processed_videos.tar    # PRD training set
PRDBench_test.tar.gz              # PRDBench test set
DragBench.tar.gz                  # DragBench test set
PRDBench_results_opt.tar          # best PRDBench results
DragBench_result.tar              # best DragBench results
ICRDrag_weights.tar.gz            # ICRDrag model weights
```

After downloading, the expected data layout is:

```text
data/
├── PRD/
│   └── train/
│       └── processed_videos/
├── PRDBench/
│   └── test/
└── DragBench/
    ├── openvid_format_dbscan_dr/
    └── openvid_format_dbscan_sr/
```

## Model Download

Model weights are available from
[[Baidu_Cloud]](https://pan.baidu.com/s/1YpNXdfqOu7fOrJYKGdb-Lw?pwd=63ue)
(access code: `63ue`).

The model weights should be restored to:

```text
weights/ICRDrag/
├── model_index.json
├── scheduler/
├── text_encoder/
├── tokenizer/
├── transformer/
└── vae/
```

## Our ICRDrag

This repository provides the PyTorch implementation, inference scripts,
evaluation scripts.

## Installation

Clone this repository:

```bash
git clone https://github.com/bcmi/ICRDrag-Region-Drag-Editing.git
cd ICRDrag-Region-Drag-Editing
```

Download the datasets, best results, and model weights from the two Baidu Cloud
links listed above, then restore them to `data/` and `weights/`.

Install dependencies:

```bash
pip install -r requirements.txt
```

`flash-attn` and `pytorch3d` are optional for this release path. Install them
only if you explicitly need FlashAttention kernels or multiview/ray generation.

## Environment

We tested the release with Python 3.10 and PyTorch CUDA wheels. A clean
environment can be created with:

```bash
conda create -n icrdrag python=3.10 pip -y
conda activate icrdrag
pip install -r requirements.txt
```

## Inference

Run commands from the repository root. Make sure `weights/ICRDrag/` and
the target benchmark data have been restored first.

### PRDBench

```bash
python3 inference/run_inference.py \
  --testset prdbench \
  --output_dir outputs/prdbench
```

The edited images are saved to:

```text
outputs/prdbench/results/
```

### DragBench-DR

```bash
python3 inference/run_inference.py \
  --testset dragbench-dr \
  --output_dir outputs/dragbench/DragBench-DR
```

### DragBench-SR

```bash
python3 inference/run_inference.py \
  --testset dragbench-sr \
  --output_dir outputs/dragbench/DragBench-SR
```

`--testset dragbench` is an alias for `dragbench-dr`. The shell wrappers under
`inference/sh_*.sh` call the same unified entrypoint.

## Evaluation

We provide evaluation scripts for PRDBench and DragBench under `evaluation/`.

### PRDBench

MSE / LPIPS / SSIM:

```bash
python3 evaluation/PRDBench/run_eval_mse_lpips_ssim.py \
  --source_dir data/PRDBench/test/target_images_rgb \
  --edited_dir outputs/prdbench/results \
  --output outputs/evaluation/similarity_results/icrdrag_results.txt
```

MD:

```bash
python3 evaluation/PRDBench/compute_MD_regiondrag.py \
  --drag_data_json data/PRDBench/test/drag_data.json \
  --source_images_dir data/PRDBench/test/source_images_rgb \
  --edited_dir outputs/prdbench/results \
  --output_dir outputs/evaluation/MD_results \
  --method_name ICRDrag \
  --sd_method sd \
  --sd_model_path weights/stable-diffusion-v1-5
```

mMD:

```bash
python3 evaluation/PRDBench/compute_mMD_draglora.py \
  --drag_data_json data/PRDBench/test/drag_data.json \
  --source_images_dir data/PRDBench/test/source_images_rgb \
  --source_masks_dir data/PRDBench/test/source_masks \
  --target_masks_dir data/PRDBench/test/target_masks \
  --edited_dir outputs/prdbench/results \
  --output_txt outputs/evaluation/mMD_results/icrdrag_mMD.txt \
  --method_name ICRDrag \
  --model_path weights/stable-diffusion-v1-5 \
  --mask_mode target \
  --seed 42
```

`compute_MD_regiondrag.py` uses the bundled RegionDrag `DragEvaluator`.
`compute_mMD_draglora.py` uses a DragLoRA-style DIFT Stable Diffusion feature
extractor in `evaluation/third_party/dift_sd.py`.
These MD/mMD scripts also require a local Stable Diffusion v1.5 feature model
at `weights/stable-diffusion-v1-5`; this external evaluator model is not part
of `ICRDrag_weights.tar.gz`.

### DragBench

DragBench evaluation scripts are grouped by split:

```text
evaluation/DragBench/DragBench_DR/
evaluation/DragBench/DragBench_SR/
```

Each folder contains scripts for MSE / LPIPS / SSIM, MD, and mMD.

## BibTeX

If you use ICRDrag, PRD, PRDBench, or the released results, please cite:

```bibtex
@article{sui2026incontext,
  title={In-context Region-based Drag: Drag Any Region to Any Shape},
  author={Sui, Jiacheng and Hao, Tianyu and Gao, Bingjie and Niu, Li and Zhai, Guangtao},
  journal={ECCV},
  year={2026}
}
```

## Acknowledgement

This implementation is built based on the
[OneDiffusion](https://github.com/OneDiffusion/OneDiffusion) codebase. We thank
the authors for their excellent open-source work.

Thanks to [Wenxuan Wu](https://github.com/wuwenxuan) for developing the Gradio demo and capturing its demo video.
