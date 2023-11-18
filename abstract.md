# Learning Representations from Remote Sensing Images

## TLDR
Transfer between remote sensing classification tasks given only a few examples.

## Motivation
Self-supervised learning can be particularly valuable for dense prediction tasks, especially given the high cost associated with gathering segmentation masks and bounding box annotations for remote sensing data.
The idea is to provide a general framework for building feature extractors with localization that work well on downstream remote sensing prediction tasks.

The abundance of unlabeled remote sensing (RS) data, characterized by its multimodal and multiscale nature, 
contrasts with the scarcity and high cost associated with labeled RS data collection. 
Can we learn good representations by leveraging this diverse pool of unlabeled data? 
Can we transfer between tasks given only a few examples?

## Data
- [EuroSAT](https://arxiv.org/pdf/1709.00029.pdf)
- [UC Merced Land Use](https://faculty.ucmerced.edu/snewsam/papers/Yang_ACMGIS10_BagOfVisualWords.pdf)
- [fMoW](https://arxiv.org/pdf/1711.07846.pdf)

## References
- [ClimaX](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2023/35/paper.pdf)
- [Lightweight, Pre-trained Transformers for Remote Sensing Timeseries](https://arxiv.org/pdf/2304.14065.pdf)
- [TIML - Task Informed Meta Learning](https://openreview.net/pdf?id=de0KufElojN)
- [Multimodal contrastive learning for remote sensing tasks](https://arxiv.org/pdf/2209.02329.pdf)
- [SSL Cookbook](https://arxiv.org/pdf/2304.12210.pdf)
- [Scale-MAE](https://arxiv.org/pdf/2212.14532.pdf)
- [SatMAE](https://arxiv.org/pdf/2207.08051.pdf)
- [Pushing the Limits of Simple Pipelines for Few-Shot Learning:
External Data and Fine-Tuning Make a Difference](https://arxiv.org/pdf/2204.07305v1.pdf)
- [Meta-Dataset](https://arxiv.org/pdf/1903.03096.pdf)
- [TIML](https://openreview.net/pdf?id=de0KufElojN)
