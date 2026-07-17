# BPNet

The Implementation of BPNet.

## 1.Datasets for Experiments

### 1.1 CSLB Dataset

Our CSLB dataset is a specialized image dataset  constructed for pixel-level segmentation of Corn Southern Leaf Blight (CSLB) lesions, aiming to support disease severity quantification and breeding-related phenotyping analysis. The details are as follows:

- **Target Task:** Precise identification and segmentation of CSLB lesions on maize leaves for lesion area quantification.
  
- **Data Format:** The dataset contains high-resolution maize leaf images paired with corresponding expert-verified pixel-level binary mask annotations. Each image is matched with one lesion mask, where lesion pixels and non-lesion pixels are represented by different labels.
  
- **Lesion Characteristics:** The dataset covers diverse CSLB infection patterns, including small scattered lesions at early infection stages and large coalesced necrotic regions at advanced stages. These variations provide a challenging benchmark for evaluating lesion segmentation models under severe class imbalance and complex lesion morphology.
  
- **Annotation Quality:** All pixel-level annotations were manually labeled and verified by researchers familiar with CSLB symptoms to ensure accurate lesion boundary delineation.
  

### 1.2 LDSD Dataset

The Leaf Disease Segmentation Dataset (LDSD) is a public benchmark dataset for pixel-level leaf lesion segmentation, covering multiple plant species, disease types, and natural imaging conditions [(LDSD)](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset). We adopt this dataset to evaluate the generalization capability of our BPNet across diverse agricultural disease scenarios.

### 1.3 ATLDSD Dataset

The Apple Tree Leaf Disease Segmentation Dataset (ATLDSD) is a public dataset for pixel-level segmentation of apple leaf disease regions under both laboratory and field imaging conditions [(ATLDSD)](https://doi.org/10.11922/sciencedb.01627). We further employ this dataset to assess the robustness of our BPNet, complementing the multi-species evaluation provided by LDSD.

## 2. Dataset Download

### 2.1 CSLB Dataset(Ours)

Our dataset is provided as a compressed `.zip` archive:

**File:** CSLB.zip

**Download Link:** https://pan.baidu.com/s/17RkT67pS99uuHdau1Ez-8Q?pwd=ydtm

**Access Code:** ydtm

### 2.2 LDSD Dataset

The public dataset is available from its original source:

https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset

### 2.3 ATLDSD Dataset

The public dataset is available from its original source: https://doi.org/10.11922/sciencedb.01627

## 3. Dataset Preparation

To ensure robust evaluation and avoid manual seed bias, five-fold cross-validation is adopted for all three datasets. Specifically, the image-mask pairs are partitioned into five subsets using Python's default seed initialization. In each fold, one subset is reserved for validation while the remaining four are used for training. Final results are reported as the mean and standard deviation across all five folds.

The final results in paper for each dataset are reported as the mean and standard deviation across the five folds.

## Training and Validation

Our BPNet is implemented based on [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

- **Training (using corn dataset as an example):**
  
  ```shell
  CUDA_VISIBLE_DEVICES=0 python train.py --name bpnet --arch BPNet --dataset corn --data_dir /BPNet --output_dir outputs --epochs 200 --batch_size 4 --input_w 1024 --input_h 1024
  ```
  
- **Validation (using corn dataset as an example):**
  
  ```shell
  CUDA_VISIBLE_DEVICES=0 python val.py --name bpnet --output_dir outputs
  ```
