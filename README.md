# BPNet

The Implementation of BPNet.

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
