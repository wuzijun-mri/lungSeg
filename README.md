# Unsupervised Segmentation of dynamic pulmonary MRI
### Introduction
This repository contains the algorithm for the research paper titled "Unsupervised segmentation of dynamic pulmonary MRI using cross-modality adaptation with annotated CT images".

A new framework was proposed for unsupervised segmentation of pulmonary MRI for lung parenchyma segmentation in dynamic pulmonary MRI by transferring segmentation knowledge from annotated CT datasets. 
The framework employs masked image modeling, incorporates temporal consistency to exploit 4D dynamic information, and develops a select-and-refine strategy for pseudo label generation.

### Train & test
1. MAE pretraining

```bash
python 1-pretrain/train.py
```

2. Train initial segmenter

```bash
python 2-initial-seg/train.py
```

3. Generate pseudo labels

```bash
python 2-initial-seg/infer.py
python 2-initial-seg/dropout.py
python 3-select-refine/Select.py
python 3-select-refine/refine.py
```

4. Train final segmenter

```bash
python 4-final-seg/train.py
```

5. Prediction

```bash
python 4-final-seg/test.py
```

We also provide a trained model and a 4D MR image from a representative subject (two respiratory phases) for demonstration purposes. Please download the model weights and data from [Zenodo](https://zenodo.org/records/18058968). To run the demo, use the following command:

```bash
python 4-final-seg/test.py --final_model=YOUR_PATH_TO_FINAL_MODEL --data_dir=YOUR_PATH_TO_TEST_DATA_DIR
```

### Acknowledgements

This repository builds upon ideas and code from multiple open-source projects, including [Pytorch-3D-UNet](https://github.com/wolny/pytorch-3dunet), [MAPSeg](https://github.com/XuzheZ/MAPSeg), [Chan-Vese Active Contour](https://github.com/kevin-keraudren/chanvese), and [DeepLab-V3-Plus](https://github.com/VainF/DeepLabV3Plus-Pytorch). 
We sincerely thank the respective authors for their valuable open-source contributions.
