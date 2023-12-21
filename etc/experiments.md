## Evaluation
For evaluating few-shot classification performance, we simulate 100 episodes/tasks from the test-split
for each dataset of interest. The evaluation metric is the average classification accuracy over tasks.
We used the convention of evaluating 5-way-1-shot (5w1s) and 5-way-5-shot (5w5s) episodes.

## Results

| ID | Architecture   | Pre-Train           | Meta-Train | OPTIMAL-31 (5w1s) | OPTIMAL-31 (5w5s) |
|----|----------------|---------------------|------------|-------------------|-------------------|
| 0  | ViT-base/16    | Sup. (ImageNet-21k) | -          | 69.4              | 90.4              |
| 1  | ViT-base/16    | DINOv2              | -          | 81.3              | 94.9              |
| 2  | ResNet-50      | Sup. (ImageNet-21k) | -          | 70.5              | 92.6              |
| 3  | ViT-base/16    | DINOv2              | ProtoNet   |                   |                   |
| 4  | ViT-large/32   | ScaleMAE (fMoW)     | -          | 45.69             | 56.93             |

