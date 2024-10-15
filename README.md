# HighwayCameraDataset
---
This repository shows how to browse and use [Highway Camera Dataset](https://huggingface.co/datasets/PrzemekS/highway-vehicles) available at Hugging Face.

### Citations
If you are using this dataset, consider citing the related paper:

`Placeholder for a paper, will appear as soon as it is published :>`

## Installation
1. Download this repository
2. Create your own conda environment, e.g:
```bash
conda create --name HighwayDataset python=3.11.5
```
3. Install pytorch. This step depends on current pytorch version, CUDA version, etc. Usually, one line of pip install is enough, and you can find it at [Pytorch get started](https://pytorch.org/get-started/locally/) website, under the `Start Locally` section.
4. Install Requirements with:
```bash
pip install -r requirements.txt
```


## What's inside
- `Browse_hf_dataset.ipnb` - A notebook that helps browsing the dataset.
- `create_yolo_dataset.py` - Code that changes HuggingFace dataset into a format acceptable for [Ultralytics](https://github.com/ultralytics/ultralytics) library.  
- `train_yolo.py` - finetunes `Yolo11n` model.
- `data_config.yaml` - required for yolo training

## How to use it
### Browsing repository
Just run view_hf_dataset.ipynb notebook. Originally it samples `NUM_IMAGES` images and displays them, but it is easy to adjust the code.

### Training yolo
1. Covert dataset to yolo format with
```bash
python create_yolo_dataset.py
```
2. Train your yolo model with
```bash
python train_yolo.py
```

## Credits
The dataset was labeled by:
- QingLian He (team leader)
- Binya Zhang
- Amir Noekhan

Code was created by [Przemek Sekula](https://github.com/PrzemekSekula)

