# DADS7202: Sport Image Classification with CNN

> **Which pre-trained CNN classifies 10 sports most accurately — and does fine-tuning the last block beat a frozen backbone?**

<p align="center"><img src="outputs/figs/1_samples.png" width="800"></p>

## Abstract
Sports image classification presents unique computer vision challenges due to rapid athlete motion, non-uniform arena illumination, and visually similar background contexts *(Podgorelec et al., 2020)*. This repository evaluates an end-to-end deep learning pipeline on a curated dataset of 7,677 images across 10 sports categories outside the standard 1,000 ImageNet classes. We compare three distinct Convolutional Neural Network (CNN) backbones (VGG16, ResNet50, and MobileNetV2), applying systematic Bayesian hyperparameter optimization via Weights & Biases (W&B) to test transfer learning (`frozen` backbone) versus fine-tuning (`last_block` unfrozen) *(Yosinski et al., 2014)*. To account for stochastic variance *(Bouthillier et al., 2021)*, the optimal configurations are evaluated across 5 random seeds, reporting overall and per-class test metrics with Welch's t-tests. Finally, model behavior is audited using Grad-CAM visual explanations *(Selvaraju et al., 2017)* and targeted error analysis.

## Key Results

1. **After W&B Sweep tuning + 5 seeds per model**, the best model is **ResNet50** with test Macro-F1 **96.64 +/- 0.19%** and accuracy **96.61 +/- 0.19%** (mean +/- SD).
2. ResNet50 vs. runner-up MobileNetV2 on Macro-F1: p = 0.0037, significant after Bonferroni (Welch's t-test, Bonferroni alpha = 0.0083).
3. The lightest model, **MobileNetV2** (2.55M params, 1.78 ms/img), reaches Macro-F1 95.34 +/- 0.54%.
4. **Fine-tuning the last block vs. a frozen backbone** (Macro-F1 difference, 95% CI): VGG16 +0.28 [-0.15, +0.70] pp; MobileNetV2 +1.99 [+1.28, +2.70] pp; ResNet50 +0.39 [+0.14, +0.63] pp.
5. Grad-CAM analysis demonstrates that our models actively focus on semantic sport cues (e.g., athletes, equipment) rather than learning spurious background artifacts.
6. Weight and Bias results: https://forge.coreweave.com/wandb/tetravaal/DADS7202_HW_SportCNN_5/reports/Untitled-Report--VmlldzoxODE3MzI1NQ?accessToken=zhdjn8pd5k8o5hi4uhlqid70uk5bcyf04ripn4h12vtwtu5fawu8yvc2yi9l5ou9

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Data](#2-data)
- [3. Network architecture](#3-network-architecture)
- [4. Training](#4-training)
- [5. Results](#5-results)
- [6. Discussion](#6-discussion)
- [7. Conclusion](#7-conclusion)
- [8. References](#8-references)

---

## 1. Introduction

- **Task:** `multi-class image classification` of sport photos into **10 classes**: badminton, basketball, boxing, chess, fencing, formula1, ice_hockey, motogp, shooting and swimming.
- **Approach:** Transfer learning from ImageNet with **3 CNN backbones of very different design**: VGG16, ResNet50 and MobileNetV2. The original classifier is replaced with a new head, and two training strategies are compared *(Yosinski et al., 2014)*:
  - `frozen`: the whole backbone is frozen and only the new head is trained (feature extraction).
  - `last_block`: the last convolutional block is unfrozen and fine-tuned together with the head using differential learning rates.
- **Tuning:** Hyperparameters are tuned with a **W&B Sweep (Bayesian)**, run separately for each architecture x strategy with the same budget.
- **Evaluation:** Each final configuration is re-trained with **5 independent seeds** to capture variance *(Bouthillier et al., 2021)*, and reported as **mean +/- SD** on a held-out test set, with Welch's t-tests between models.
- **Interpretation:** We use **Grad-CAM** *(Selvaraju et al., 2017)*, eyeball analysis, and error analysis to check *why* the models decide as they do.

---

## 2. Data

### 2.1 Data source

All images were collected from the web and pre-filtered with CLIP before entering the notebook:

1. **Collection:** Images were downloaded with **icrawler** from **Bing** (photo filter on) and **Baidu**, using about 10 keywords per class (up to 100 images per keyword per engine). The keywords mix several angles and contexts (match, action shot, venue) and several languages (English, Thai, Chinese, Japanese, Spanish, Russian), so that no single source dominates. Corrupt files, images with a short side < 64 px, and pHash duplicates were then removed. A duplicate that appeared in two classes was dropped entirely.
2. **Relevance filtering:** **`openai/clip-vit-large-patch14`** (CLIP ViT-L/14) was used zero-shot as a flagging tool. Each image was scored against 3 prompt templates per sport (e.g., "an action shot of {sport}") and 12 "junk" prompts (cartoon, product photo, empty stadium, logo/text, video game). An image was flagged as **junk** if p(junk) >= 0.60, or as **mismatch** if CLIP predicted another sport with p >= 0.80 while the given label had p <= 0.10. Flagged images were **checked by eye**, and only those confirmed as wrong were moved out (not deleted). The same prompts and thresholds were used for every image, before splitting.
3. **Duplicate removal:** **`openai/clip-vit-base-patch32`** (CLIP ViT-B/32) image embeddings were compared by cosine similarity. Pairs with similarity > 0.95 were listed as near-duplicates for removal. This catches re-crops and edits that pHash misses.

The notebook then applies its own cleaning (Section 2.3), with MD5 for exact copies and dHash for resized copies across splits.

| Class | Source(s) | # images (after cleaning) | Note |
|---|---|:-:|---|
| badminton | Web (icrawler) - Bing/Baidu | 771 | Indoor court action |
| basketball | Web (icrawler) - Bing/Baidu | 817 | Indoor/outdoor courts |
| boxing | Web (icrawler) - Bing/Baidu | 691 | High-contrast spotlighting |
| chess | Web (icrawler) - Bing/Baidu | 904 | Close-up board views |
| fencing | Web (icrawler) - Bing/Baidu | 870 | Dark piste backgrounds |
| formula1 | Web (icrawler) - Bing/Baidu | 671 | High-speed circuit racing |
| ice_hockey | Web (icrawler) - Bing/Baidu | 770 | High-brightness rink surfaces |
| motogp | Web (icrawler) - Bing/Baidu | 795 | Motorcycles on tracks |
| shooting | Web (icrawler) - Bing/Baidu | 686 | Stance and aiming profiles |
| swimming | Web (icrawler) - Bing/Baidu | 702 | Water splashes, pool lanes |

### 2.2 Exploratory data analysis

<p align="center"><img src="outputs/figs/1_eda.png" width="850"></p>

After cleaning: **7,677 images**; largest class **chess** (904), smallest **formula1** (671) -> imbalance ratio (max/min) = **1.35**.

To handle class imbalance and prevent the model from biasing toward majority classes *(Cui et al., 2019)*, we:
1. Use a **stratified split**, so every split keeps the same class proportions.
2. Use **class-weighted cross-entropy**, with weights calculated from the train split only.
3. Report imbalance-robust metrics: **Macro-F1** (primary), balanced accuracy and per-class precision/recall.

Additionally, the EDA reveals significant illumination variations (e.g., indoor vs. outdoor sports), justifying the need for photometric augmentation to handle low-light and high-contrast environments *(Guo et al., 2020)*.

### 2.3 Data cleaning

As demonstrated by *Northcutt et al. (2021)*, pervasive label errors and cross-set leakage destabilize machine learning benchmarks. We implement a systematic dataset curation step:
1. Remove unreadable or corrupt files.
2. Remove images whose short side is < 64 px.
3. Remove exact duplicates (same MD5). If the same file appears in two classes, every copy is dropped because the labels conflict (resolving dataset-level label noise).
4. Remove near-duplicates, i.e., resized or re-encoded copies, using a perceptual hash (dHash, Hamming distance <= 4).
5. Fix the EXIF orientation, convert to RGB, and downscale so the short side is <= 512 px.

A **near-duplicate leakage check** then compares the dHash of every image across train/val/test, because MD5 cannot detect a resized copy of the same photo:

> **4 near-duplicate pair(s) across splits** (1 involving the test set). They were **kept** (`REMOVE_NEAR_DUPLICATES = False`) and are reported as a limitation: all pairs have the same label, and 1 of the 1,152 test images has a near-copy in another split, so the effect on test accuracy is at most 1/1,152 = 0.09 pp.

<p align="center"><img src="outputs/figs/2_near_duplicates.png" width="500"></p>

This is well below the 1.30 pp gap between ResNet50 and MobileNetV2 (Section 5.4), so the leak cannot change the ranking of the models.

### 2.4 Data splitting (train / val / test)

- **Stratified random split 70 / 15 / 15** (`random_state = 42`).
- The **same split is used for every architecture and seed**, so the run-to-run variation comes only from training randomness.
- The test set is used **only for the final evaluation**. Hyperparameters, the best epoch and the strategy are all chosen on validation.

| Class | train | val | test | total |
|---|:-:|:-:|:-:|:-:|
| badminton | 539 | 116 | 116 | 771 |
| basketball | 572 | 123 | 122 | 817 |
| boxing | 484 | 104 | 103 | 691 |
| chess | 633 | 135 | 136 | 904 |
| fencing | 609 | 130 | 131 | 870 |
| formula1 | 470 | 100 | 101 | 671 |
| ice_hockey | 539 | 116 | 115 | 770 |
| motogp | 556 | 119 | 120 | 795 |
| shooting | 480 | 103 | 103 | 686 |
| swimming | 491 | 106 | 105 | 702 |
| **Total** | 5373 | 1152 | 1152 | 7677 |

### 2.5 Pre-processing & data augmentation

| | Train (online augmentation) | Val / Test |
|---|---|---|
| Geometry | `RandomResizedCrop(224, scale=(0.5, 1))`, `RandomHorizontalFlip(0.5)`, `RandomRotation(+/-10 degrees)` | `Resize(256)` -> `CenterCrop(224)` |
| Photometric | `ColorJitter(brightness 0.3, contrast 0.3, saturation 0.2, hue 0.02)` | -- |
| Normalization | ImageNet mean / std | same |

Sport photos vary in camera angle, zoom and lighting, so we use scale crops, horizontal flips, small rotations and colour jitter. Vertical flips and large rotations are not used because they produce unnatural images.

<p align="center"><img src="outputs/figs/2_augmentation.png" width="800"></p>

---

## 3. Network architecture

### 3.1 Pre-trained backbones

We selected three distinct CNN families to evaluate trade-offs between representational depth, parameter footprint, and inference speed:

| Backbone | Year | Params | Key idea | Why chosen |
|---|:-:|:-:|---|---|
| **VGG16** *(Simonyan & Zisserman, 2015)* | 2014 | ~138M (original) | Plain stack of 3x3 convs (13 conv layers) | Classic baseline; provides straightforward spatial feature maps for Grad-CAM. |
| **ResNet50** *(He et al., 2016)* | 2015 | ~25.6M | Residual (skip) connections + bottleneck blocks | Deep yet easy to train; a benchmark standard for high-accuracy visual recognition. |
| **MobileNetV2** *(Sandler et al., 2018)* | 2018 | ~3.5M | Depthwise-separable convs + inverted residuals | Light and fast; suitable for latency-sensitive or resource-constrained devices. |

### 3.2 Pre-trained models without fine-tuning

Before any training, the original ImageNet-1K models predict standard ImageNet classes (e.g., `racket`, `puck`, `scuba_diver`), and none of their outputs corresponds directly to our 10 sport classes. The classifier therefore must be replaced and fine-tuned on domain-specific data.

<p align="center"><img src="outputs/figs/3_pretrained_resnet50.png" width="800"></p>

The figure above demonstrates that the native, un-tuned ResNet50 model completely fails to capture the correct sports context, incorrectly returning generic equipment or irrelevant environmental tags rather than the core sport.

### 3.3 Modified architecture (backbone cut + new head)

Following transfer learning principles *(Yosinski et al., 2014)*, the convolutional feature extractor of each backbone is retained while its original 1,000-class output layer is replaced by a unified 2-layer MLP classification head.

| Backbone | Kept (ImageNet weights) | Removed | New head | Unfrozen in `last_block` |
|---|---|---|---|---|
| VGG16 | `features` + `avgpool` -> 25,088-d | `classifier` (FC 4096->4096->1000) | `Linear(25088, H) -> ReLU -> Dropout(p) -> Linear(H, 10)` | `features[24:]` (conv block 5) |
| ResNet50 | conv1 ... layer4 + avgpool -> 2,048-d | `fc` (2048->1000) | `Linear(2048, H) -> ReLU -> Dropout(p) -> Linear(H, 10)` | `layer4` |
| MobileNetV2 | `features` + global avgpool -> 1,280-d | `classifier` (1280->1000) | `Linear(1280, H) -> ReLU -> Dropout(p) -> Linear(H, 10)` | `features[14:]` |

The **same head structure** is used for all 3 backbones, so only the representation of the backbones differs. `H` and `p` are hyperparameters tuned by the sweep. BatchNorm layers in the frozen part stay in `eval()` mode, ensuring the ImageNet running statistics are not corrupted during training.

<p align="center"><img src="outputs/figs/3_network_diagram.png" width="900"></p>

---

## 4. Training

### 4.1 Training setup

| Item | Setting |
|---|---|
| Loss | Class-weighted cross-entropy (weights derived from the train split to counteract imbalance) |
| Optimizer | AdamW: head LR = `lr`, LR of the unfrozen backbone block = `lr x 0.1` |
| LR scheduler | CosineAnnealingLR (T_max = epochs) |
| Epochs / batch size | 12 / 32 (fixed) |
| Model selection | Checkpoint of the epoch with the highest **val macro-F1** |
| Hardware | Mac mini (Apple M4: 10-core CPU, 10-core GPU, 16 GB), PyTorch MPS backend. About 20.0 h of training in total: 13.3 h for the 60 sweep runs and 6.7 h for the 30 final runs. |

### 4.2 Hyperparameter tuning: W&B Sweep (Bayesian)

To compare `frozen` and `last_block` strategies fairly, **one sweep is run per architecture per strategy** (6 sweeps) with the same budget of **10 runs each**, rather than letting a single sweep choose the strategy. The objective metric is to maximize the best validation macro-F1 (using a single fixed seed = 42).

| Parameter | Search space |
|---|---|
| `lr` | log-uniform [1e-4, 2e-3] |
| `weight_decay` | {1e-4, 1e-2} |
| `hidden` (H) | {256, 512} |
| `dropout` (p) | {0.3, 0.5} |

**Best configuration per architecture x strategy** (from the sweeps):

| Model | Strategy | lr | weight_decay | hidden (H) | dropout (p) |
|---|:-:|:-:|:-:|:-:|:-:|
| VGG16 | `frozen` | 1.28e-04 | 0.0001 | 512 | 0.5 |
| VGG16 | `last_block` (Selected) | 1.50e-03 | 0.0001 | 512 | 0.3 |
| MobileNetV2 | `frozen` | 1.52e-03 | 0.01 | 512 | 0.3 |
| MobileNetV2 | `last_block` (Selected) | 1.70e-03 | 0.0001 | 256 | 0.5 |
| ResNet50 | `frozen` | 1.05e-03 | 0.01 | 512 | 0.3 |
| ResNet50 | `last_block` (Selected) | 9.84e-04 | 0.0001 | 512 | 0.5 |

> (Selected) indicates the strategy chosen as the main model due to a higher mean validation macro-F1 over the 5 seeds.

**Sweep results** (6 sweeps x 10 runs = 60 runs; best val macro-F1 in %, single seed 42):

| Model | Strategy | Runs | Best val macro-F1 | Median | Worst | Avg. time / run (min) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| VGG16 | `frozen` | 10 | **93.36** | 93.06 | 92.39 | 17.0 |
| VGG16 | `last_block` | 10 | **94.89** | 94.13 | 93.40 | 19.1 |
| MobileNetV2 | `frozen` | 10 | **94.69** | 94.51 | 91.46 | 7.1 |
| MobileNetV2 | `last_block` | 10 | **96.86** | 96.65 | 95.05 | 7.9 |
| ResNet50 | `frozen` | 10 | **97.88** | 97.61 | 96.27 | 12.9 |
| ResNet50 | `last_block` | 10 | **98.13** | 97.83 | 96.94 | 15.6 |

**What the sweeps show:**
- **The learning rate matters most.** For MobileNetV2 and ResNet50, val macro-F1 rises with lr (Pearson r between log lr and val F1 = 0.82-0.95 across the 4 sweeps), and the best runs sit near the top of the range. For VGG16 there is no clear trend (r = -0.27 and 0.02).
- `hidden = 512` gives a higher mean val F1 than 256 in 5 of the 6 sweeps. The effects of `weight_decay` and `dropout` are small (mostly < 0.5 pp) and change direction between sweeps.
- The Bayesian search samples unevenly, with some values tried only 1-2 times, so these trends are indicative only.

### 4.3 Final runs and strategy selection

To account for stochastic optimization variance and initialization noise *(Bouthillier et al., 2021)*:
1. The best configuration of **both strategies** is re-trained from scratch with **5 seeds** (`0-4`).
2. For each architecture, the strategy with the higher **mean val macro-F1** becomes the main model. The choice is made on validation only, never on test.
3. Results are reported on the test set as mean +/- SD over the 5 seeds.

**Final runs:** 3 models x 2 strategies x 5 seeds = 30 runs, all finished. Values are mean +/- SD over seeds 0-4. (Selected) marks the strategy with the higher mean **val** macro-F1, which becomes the main model.

| Model | Strategy | Runs | Val macro-F1 (%) | Test macro-F1 (%) | Test accuracy (%) | Avg. time / run (min) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| VGG16 | `frozen` | 5 | 92.81 +/- 0.22 | 92.48 +/- 0.35 | 92.50 +/- 0.36 | 17.3 |
| VGG16 | `last_block` (Selected) | 5 | 94.03 +/- 0.25 | 92.76 +/- 0.15 | 92.80 +/- 0.14 | 19.4 |
| MobileNetV2 | `frozen` | 5 | 94.73 +/- 0.43 | 93.34 +/- 0.42 | 93.37 +/- 0.43 | 7.1 |
| MobileNetV2 | `last_block` (Selected) | 5 | 96.68 +/- 0.31 | 95.34 +/- 0.54 | 95.31 +/- 0.53 | 7.9 |
| ResNet50 | `frozen` | 5 | 97.56 +/- 0.14 | 96.25 +/- 0.15 | 96.20 +/- 0.16 | 13.1 |
| ResNet50 | `last_block` (Selected) | 5 | 97.81 +/- 0.30 | 96.64 +/- 0.19 | 96.61 +/- 0.19 | 15.8 |

- `last_block` wins on validation for all 3 models, and the test set agrees in every case.
- Test macro-F1 is **below val** for every configuration: about 1.2-1.4 pp for most, and 0.3 pp for VGG16 `frozen`. Val was used to pick the hyperparameters, the best epoch and the strategy, so it is optimistically biased, which is why only test scores are reported as results. The ranking of the models is the same on val and test.

---

## 5. Results

### 5.1 Test performance (mean +/- SD over 5 seeds)

| Model | Strategy | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Weighted F1 (%) | Top-3 Acc (%) | Params (M) | Trainable (M) | Latency (ms/img) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **ResNet50** | last_block | 96.61 +/- 0.19 | 96.64 +/- 0.20 | 96.64 +/- 0.19 | 96.62 +/- 0.19 | 99.34 +/- 0.32 | 24.56 | 16.02 | 5.84 |
| **MobileNetV2** | last_block | 95.31 +/- 0.53 | 95.35 +/- 0.52 | 95.34 +/- 0.54 | 95.34 +/- 0.52 | 99.10 +/- 0.17 | 2.55 | 2.01 | 1.78 |
| **VGG16** | last_block | 92.80 +/- 0.14 | 92.70 +/- 0.18 | 92.76 +/- 0.15 | 92.82 +/- 0.12 | 98.56 +/- 0.21 | 27.57 | 19.93 | 8.75 |

> Ranked by mean test Macro-F1 (primary metric). SD = sample SD (ddof = 1). Latency = forward-pass time per image at batch size 32 (throughput, not single-image latency), measured on the training machine (Apple Silicon, MPS).

### 5.2 Training curves

<p align="center"><img src="outputs/figs/6_training_curves.png" width="900"></p>

> Train metrics are measured with augmentation and dropout on. At the final epoch, train accuracy exceeds validation accuracy by +5.1 pp (VGG16), +2.2 pp (MobileNetV2) and +2.1 pp (ResNet50). VGG16 overfits the most, likely because its new head takes the 25,088-d flattened feature map (~12.8M params in the first Linear layer).

### 5.3 Per-class performance & confusion matrices

<p align="center"><img src="outputs/figs/6_perclass_f1.png" width="850"></p>
<p align="center"><img src="outputs/figs/6_confusion.png" width="900"></p>

**Per-class F1 (%), mean +/- SD**

| Class | VGG16 | MobileNetV2 | ResNet50 |
|---|:-:|:-:|:-:|
| badminton | 87.29 +/- 1.58 | 91.20 +/- 0.93 | 93.86 +/- 0.94 |
| basketball | 86.86 +/- 1.26 | 92.21 +/- 1.34 | 92.25 +/- 1.15 |
| boxing | 91.34 +/- 1.32 | 94.70 +/- 0.79 | 94.56 +/- 0.57 |
| chess | 95.81 +/- 0.84 | 96.01 +/- 0.24 | 97.59 +/- 0.48 |
| fencing | 94.93 +/- 1.28 | 96.35 +/- 0.52 | 97.37 +/- 0.19 |
| formula1 | 92.99 +/- 1.78 | 95.76 +/- 1.40 | 98.73 +/- 0.89 |
| ice_hockey | 91.01 +/- 0.85 | 96.02 +/- 0.34 | 97.28 +/- 0.65 |
| motogp | 98.74 +/- 0.50 | 98.30 +/- 0.96 | 98.99 +/- 0.49 |
| shooting | 91.92 +/- 0.98 | 95.21 +/- 0.43 | 97.10 +/- 0.36 |
| swimming | 96.71 +/- 0.54 | 97.60 +/- 0.34 | 98.68 +/- 0.61 |

- Weakest class for VGG16: **basketball** (86.86 +/- 1.26%)
- Weakest class for MobileNetV2: **badminton** (91.20 +/- 0.93%)
- Weakest class for ResNet50: **basketball** (92.25 +/- 1.15%)

### 5.4 Statistical significance between architectures

To evaluate whether performance differences are statistically significant *(Bouthillier et al., 2021)*, we apply Welch's t-test (two-sided) on test accuracy and Macro-F1 over the 5 seeds. A Bonferroni correction for 3 pairs x 2 metrics = 6 tests gives a threshold of **alpha = 0.05 / 6 = 0.0083**.

| metric | pair | diff (pp) | p-value | significant (Bonferroni) |
|---|:-:|:-:|:-:|:-:|
| Accuracy | VGG16 vs MobileNetV2 | -2.52 | 0.0003 | Yes |
| Accuracy | VGG16 vs ResNet50 | -3.82 | 0.0000 | Yes |
| Accuracy | MobileNetV2 vs ResNet50 | -1.30 | 0.0035 | Yes |
| Macro F1 | VGG16 vs MobileNetV2 | -2.58 | 0.0002 | Yes |
| Macro F1 | VGG16 vs ResNet50 | -3.88 | 0.0000 | Yes |
| Macro F1 | MobileNetV2 vs ResNet50 | -1.30 | 0.0037 | Yes |

### 5.5 Ablation: frozen backbone vs. fine-tuning the last block

The mean difference is `last_block - frozen` on the test set, with a Welch 95% CI, a Bonferroni-adjusted p-value (x 3 architectures) and Cohen's d.

| metric | model | frozen (mean +/- SD) [95% CI] % | last_block (mean +/- SD) [95% CI] % | diff LB-FZ (pp) [95% CI] | p (Welch) |
|---|:-:|:-:|:-:|:-:|:-:|
| test macro-F1 | VGG16 | 92.48 +/- 0.35 [92.05, 92.91] | 92.76 +/- 0.15 [92.58, 92.95] | +0.28 [-0.15, +0.70] | 0.1553 |
| test macro-F1 | MobileNetV2 | 93.34 +/- 0.42 [92.82, 93.87] | 95.34 +/- 0.54 [94.67, 96.00] | +1.99 [+1.28, +2.70] | 0.0002 |
| test macro-F1 | ResNet50 | 96.25 +/- 0.15 [96.07, 96.43] | 96.64 +/- 0.19 [96.41, 96.87] | +0.39 [+0.14, +0.63] | 0.0070 |

<p align="center"><img src="outputs/figs/7x_strategy_diff.png" width="800"></p>

### 5.6 Grad-CAM analysis

Grad-CAM *(Selvaraju et al., 2017)* is computed on the **best seed of each architecture, chosen by val macro-F1**, with target = the predicted class. The top rows show images all models classify correctly. The bottom rows show images at least one model gets wrong.

<p align="center"><img src="outputs/figs/7_gradcam.png" width="900"></p>

**Findings:**
The heatmaps confirm that the fine-tuned models actively focus on semantic sport cues such as athletes, rackets, and vehicles, rather than memorizing spurious background artifacts like stadium banners or watermarks. The three architectures generally focus on similar regions, though ResNet50 exhibits tighter localization. On misclassified images, the models occasionally attend to ambiguous shared features, such as indoor lighting rigs or generic court lines.

### 5.7 Eyeball analysis: pre-trained vs. fine-tuned models

<p align="center"><img src="outputs/figs/7_eyeball.png" width="900"></p>

The visual comparison illustrates that while the original ImageNet models predict generic objects (e.g., classifying an ice hockey player simply as a 'puck' or 'crash helmet'), the fine-tuned models correctly identify the overall sport context, successfully bridging the domain gap.

### 5.8 Error analysis

<p align="center"><img src="outputs/figs/7_error_analysis.png" width="900"></p>

ResNet50's weakest class is basketball (92.25%), which is often confused with other indoor court sports due to shared features like wooden floors and similar spectator backgrounds. Similar trends are observed in VGG16 (weakest: basketball) and MobileNetV2 (weakest: badminton).

---

## 6. Discussion

- **Which architecture works best:** ResNet50 > MobileNetV2 > VGG16 on test Macro-F1 (96.64 / 95.34 / 92.76%), and all three pairwise gaps are significant after Bonferroni (5.4). The residual design lets ResNet50 learn deep features that transfer well. VGG16's plain stack ends in a 25,088-d flattened map, which makes the head large and prone to overfitting (train-val gap +5.1 pp vs. ~+2 pp for the others).
- **Fine-tuning does not help equally:** unfreezing the last block helps MobileNetV2 the most (+1.99 pp, d = 4.1), helps ResNet50 a little (+0.39 pp) and gives no significant gain for VGG16 (+0.28 pp, p_adj = 0.47). MobileNetV2's last block includes the final 1x1 conv that produces the 1,280-d features, so unfreezing it adapts the feature space directly. ResNet50's frozen features are already strong (96.25%), which leaves less room to improve.
- **Accuracy vs. size vs. speed:** MobileNetV2 has about 10x fewer parameters (2.55M vs. 24.56M) and is about 3.3x faster (1.78 vs. 5.84 ms/img), for 1.30 pp lower Macro-F1. That makes it the better fit for a mobile or on-device app, while ResNet50 suits a server where accuracy matters most. VGG16 is the largest and slowest model and also the least accurate, so it has no use case here.
- **Possible under-training:** the best validation epoch often falls at the end of the 12-epoch budget, for MobileNetV2 `last_block` (best epochs 12, 7, 11, 7, 12) and especially VGG16 `frozen` (12, 12, 12, 11, 10). Validation F1 was likely still improving, so a longer schedule (e.g. 20-30 epochs) could raise these scores. The VGG16 `frozen` result may therefore be underestimated, which could partly explain why its ablation gap is not significant.
- **Fair comparison by design:** each strategy got its own sweep with the same budget, and the main model's strategy was selected on validation only. The test set never influenced any choice.
- **Limitations:**
  - With n = 5 seeds, statistical power is low.
  - The p-values and CIs reflect only training randomness (initialization, batch order, augmentation), not variation from sampling the data, because the split is fixed and each sweep used a single seed.
  - 10 sweep runs per architecture x strategy may not find the optimum.
  - Several selected learning rates lie near the edge of the search range [1e-4, 2e-3]: VGG16 `frozen` 1.28e-4 sits near the lower bound, while VGG16 `last_block` 1.50e-3, MobileNetV2 `last_block` 1.70e-3 and MobileNetV2 `frozen` 1.52e-3 sit near the upper bound. The optimum may lie outside the range, so a wider search (e.g. [3e-5, 1e-2]) is left for future work. The unfrozen block is trained at only `lr x 0.1` (<= 2e-4), which may also limit how much fine-tuning can adapt the backbone.

---

## 7. Conclusion

- We compared 3 pre-trained CNNs (VGG16, ResNet50, MobileNetV2) on a 10-class sport dataset, each tuned with its own W&B Sweep per strategy.
- Ranking by mean test Macro-F1: ResNet50 (96.64 +/- 0.19%) > MobileNetV2 (95.34 +/- 0.54%) > VGG16 (92.76 +/- 0.15%).
- Best model: **ResNet50** (last_block); ResNet50 vs. MobileNetV2: p = 0.0037, significant after Bonferroni.
- Strategy chosen on validation per model: ResNet50 -> `last_block`, MobileNetV2 -> `last_block`, VGG16 -> `last_block`.
- Grad-CAM shows that feature extraction appropriately targets primary subjects rather than background artifacts.
- Error analysis shows that the most frequent errors occur between classes that share substantial environmental backgrounds (e.g., indoor courts).

### Future work
- Collect more images or hard examples for the weakest class, unfreeze more layers, train for more epochs, and try newer backbones (EfficientNet, ConvNeXt, ViT).
- Implement automated label noise detection to further sanitize the web-scraped dataset prior to training.

---

## 8. References

**Libraries:** PyTorch, torchvision, scikit-learn, SciPy, pandas, NumPy, Matplotlib, Weights & Biases, pytorch-grad-cam, torchinfo, Pillow, icrawler (data collection), CLIP via Hugging Face (data filtering)

**Versions:** Python 3.10.21 | PyTorch 2.5.1 | torchvision 0.20.1

1. Bouthillier, X., et al. (2021). **[Accounting for variance in machine learning benchmarks](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)**. Proceedings of Machine Learning and Systems (MLSys), 3, 747-769.
2. Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). **[Class-balanced loss based on effective number of samples](https://arxiv.org/abs/1901.05555)**. CVPR.
3. Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., & Cong, R. (2020). **[Zero-reference deep curve estimation for low-light image enhancement](https://arxiv.org/abs/2001.06826)**. CVPR.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**. CVPR.
5. Northcutt, C. G., Athalye, A., & Mueller, J. (2021). **[Pervasive label errors in test sets destabilize machine learning benchmarks](https://arxiv.org/abs/2103.14749)**. NeurIPS.
6. Podgorelec, V., Pecnik, S., & Vrbancic, G. (2020). **[Classification of similar sports images using convolutional neural network with hyper-parameter optimization](https://doi.org/10.3390/app10238494)**. Applied Sciences.
7. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). **[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)**. CVPR.
8. Selvaraju, R. R., et al. (2017). **[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)**. ICCV.
9. Simonyan, K., & Zisserman, A. (2015). **[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)**. ICLR.
10. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). **[How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)**. NeurIPS.

