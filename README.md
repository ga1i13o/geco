# 🦎 GECO: Geometrically consistent embedding with lightspeed inference

This repository contains the official implementation of:

**GECO: Geometrically consistent embedding with lightspeed inference** 

[ 🌐**Project Page**](https://reginehartwig.github.io/publications/geco/) • [📄  **Paper**](TBA)
<div style="display: flex; align-items: center; gap: 10px;">
    <img src="assets/src_10_2_crop.png" alt=" " width="200">
    <img src="assets/trg_10_2_crop_blend.gif" alt=" " width="200">
</div>
</p>
<div style="display: flex; align-items: center; gap: 10px;">
<img src="assets/src_01_5_crop.png" alt=" " width="200">
<img src="assets/trg_01_5_crop_blend.gif" alt=" " width="200">
</p>
</div>


## 🔧 Environment Setup
If you're using a Linux machine, set up the Python environment with:
```bash
conda create --name geco python=3.10
conda activate geco
bash setup_env.sh
```
To use [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) for mask extraction:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

```

## 🚀 Get Started

### 📁  Prepare the Datasets

Run the following scripts to prepare each dataset:
* APK:
    ```bash
    bash download_data/prepare_apk.sh
    wget https://github.com/Junyi42/GeoAware-SC/blob/master/prepare_ap10k.ipynb
    ```
    Then run the notebook from GeoAware-SC to preprocess the data.
* CUB:
    ```bash
    bash download_data/prepare_cub.sh
    ```
* PascalParts
    ```bash
    bash download_data/prepare_pascalparts.sh
    ```
* PFPascal
    ```bash
    bash download_data/prepare_pfpascal.sh
    ```
* SPair-71k:
    ```bash
    bash download_data/prepare_spair.sh
    ```

### Extract the mask
1. Define `<your_precomputed_masks_path>` in the dataset config files.
2. Define `<path-model-seg>` in `store_masks.yaml` pointing to the path, where `sam_vit_h_4b8939.pth`is stored.
3. Select the datasets to process in `store_masks.yaml`.
4. Run:
   ```bash
    python scripts/store_masks.py --config-name=store_masks.yaml 
    ```

### Precompute the features (not recommended)
Choose a path and define `<your_precomputed_feats_path>` in the dataset config files.
Define which dataset you want to extract the features for in `store_feats.yaml` and run
```bash
python scripts/store_feats.py --config-name=store_feats.yaml 
```


## 🎯 Pretrained Weights
Download pretrained weights:
    ```bash
    bash download_data/pretrained_weights.sh
    ```


## 🧪 Interactive Demos: Give it a Try!

We provide interactive jupyter notebooks for testing.

* [📚 Data Loading Demo](demo_data_loading.ipynb)

    Validate dataset preparation and path setup.

* [🎨 Segmentation Demo](demo_segmenation_nearest_centroid.ipynb)

    Visualize part segmentation using a simple linear classifier.

* [📍 Keypoint Transfer Demo](demo_keypoint_transfer.ipynb)

    Explore keypoint transfer and interactive attention maps.


## 📊 Run Evaluation

Run full evaluation:
```bash
python scripts/run_evaluation_all.py --config-name=eval_all.yaml 
```
Evaluate inference time and memory usage:
```bash
python scripts/run_evaluation_time_mem.py --config-name=eval_time_mem.yaml 
```
Evaluate segmentation metrics:
```bash
python scripts/run_evaluation_seg.py --config-name=eval_seg.yaml 
```


## 🏋️ Train the Model
Before training, comment out the following block in `configs/featurizer/dinov2lora.yaml`:

```yaml
init:
  id: geco
  eval_last: True
```
Then run:
```bash
python scripts/train_pairs.py --config-name=train_pairs.yaml 
```