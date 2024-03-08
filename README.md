## PyTorch implementation of the paper "Spectral U-Net: Enhancing Medical Image Segmentation via Spectral Decomposition"


set the nnUNet environmental variable

```python
export nnUNet_raw=/path/to/raw_data
export nnUNet_preprocessed=/path/to/preprocessed_data
export nnUNet_results=/path/to/result
```

### 1. Retina Fluid Segmentation

1. download the dataset from ![https://retouch.grand-challenge.org/Download/](https://retouch.grand-challenge.org/Download/)

2. run `python nnunetv2/run/run_training.py dataset_id configuration fold -tr RetinaTrainer --no-debug` for training and testing. Please refer to [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for details about `dataset_id`, `configuration` and `fold`.

### 2. BraTS Segmentation

1. download the datast from ![Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
2. run `python nnunetv2/run/run_training.py dataset_id configuration fold -tr BrainTrainer --no-debug` for training and testing.

### 3. LiTS Segmentation

1. download the datast from ![Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
2. run `python nnunetv2/run/run_training.py dataset_id configuration fold -tr LiverTrainer --no-debug` for training and testing.
