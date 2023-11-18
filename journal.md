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