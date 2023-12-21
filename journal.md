### 9.19.2023
- Opened project repo, initial commit and general project structure.

### 10.18.2023
- Literature review on SSL for dense prediction tasks and incorporating multiple modalities.

### 10.30.2023
- Read more recent papers on SSL on remote sensing data [Scale-MAE](https://arxiv.org/pdf/2212.14532.pdf), [SatMAE](https://arxiv.org/pdf/2207.08051.pdf).
- Chose 3-5 remote sensing image classification datasets from [EarthNets](https://arxiv.org/pdf/2210.04936.pdf).

### 11.6.2023
- Read papers on meta-learning algorithms [MAML](https://arxiv.org/pdf/1703.03400.pdf), [ProtoNets](https://arxiv.org/pdf/1703.05175.pdf) and a recent paper [P>M>F](https://arxiv.org/pdf/2204.07305v1.pdf) that suggest a simple pipeline that incorporates a self-supervised pre-trained backbone prior to the meta-training of the feature backbone and performs well on [Meta-Dataset](https://arxiv.org/pdf/1903.03096.pdf).
- Read [TIML](https://openreview.net/pdf?id=de0KufElojN) which offers a task-informed meta-learning algorithm, that takes relevant task-specific metadata and uses it to augment the learning proces. TIML was evaluated on the CropHarvest dataset, a global dataset of agricultural
class labels paired with remote sensing data.

### 11.17.2023
- Implemented BaseImageDataset and initial ProtoNet training script.
- Currently experimenting with these 3 datasets: [EuroSAT](https://arxiv.org/pdf/1709.00029.pdf), [UC Merced Land Use](https://faculty.ucmerced.edu/snewsam/papers/Yang_ACMGIS10_BagOfVisualWords.pdf), [fMoW](https://arxiv.org/pdf/1711.07846.pdf).

### 11.20.2023
- Read DINO + DINOv2 papers

### 11.29.2023
- Implemented dataset splits according to the common Few-shot scheme.
- Decided on using pre-trained models + K-NN on test and validation episodes at first. Second, fine-tune on meta-test only.

### 11.30.2023
- Implemented pre-trained models as baselines.
- Read the [fMoW](https://arxiv.org/pdf/1711.07846.pdf) paper, thinking about GSD and multi-spectral augmentations, but still TBD.

### 12.1.2023
- Started running experiments, currently without meta-training, only examining backbones and pre-training regimes on FS evaluation conventions. Refer to `etc/experiments.md`.

### 12.3.2023
- Ran experiments on different pre-trained models. SSL (DinoV2) yields very good results compared to the same architecture with fully-supervised on ImageNet.

### 12.17.2023
- Used [Scale-MAE](https://arxiv.org/pdf/2212.14532.pdf) as a pre-trained "foundation model" for satellite images. Results are underwhelming even after trying multiple tweaks. Perhaps some comments on how to reproduce features for each dataset (OPTIMAL-31 was evaluated in the paper) could have helped in obtaining better results.    