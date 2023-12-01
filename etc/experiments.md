## Evaluation
For evaluating few-shot classification performance, we simulate 100 episodes/tasks from the test-split
for each dataset of interest. The evaluation metric is the average classification accuracy over tasks.
We used the convention of evaluating 5-way-1-shot (5w1s) and 5-way-5-shot (5w5s) episodes.

## Results

| ID | Architecture | Pre-Train           | Meta-Train | OPTIMAL-31 (5w1s) | OPTIMAL-31 (5w1s) |
|----|--------------|---------------------|------------|-------------------|-------------------|
| 0  | ViT-base/16  | Sup. (ImageNet-21k) | -          | 69.36             | 90.37             |
| 1  | ViT-base/16  | DINOv2              | -          |                   |                   |
| 2  | ResNet-50    | Sup. (ImageNet-21k) | -          |                   |                   |
