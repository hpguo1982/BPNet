# BPNet

The Implementation of BPNet.

## The CSLB Dataset

The **CSLB Dataset** is a specialized image collection focused on Southern Corn Leaf Blight (CSLB), designed to support **phenotypic quantification** and **breeding selection** workflows.

- **Target:** Precise identification and measurement of CSLB lesions on maize leaves.

- **Data Format:** High-resolution original images paired with pixel-level mask annotations for clear boundary definition.

- **Morphological Range:** Includes various infection stages, from early-stage spots to severe necrotic areas, enabling models to calculate disease density and leaf coverage ratios. 

The files are stored in a `.zip` archive on Baidu Network Disk:

- **Link:** [Baidu Network Disk (CSLB.zip)](https://pan.baidu.com/s/17RkT67pS99uuHdau1Ez-8Q?pwd=ydtm)

- **Access Code:** `ydtm`

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
