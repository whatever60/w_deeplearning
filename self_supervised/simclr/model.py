'''
- Data augmentation
- None-linear projection head
- Tempreture and normalization of the output
- Large batch sizes and negative sample number
- NT-Xnet Loss
- Global batch-normalization

Data augmentation:
- Random crop + resize + horizontal flip
- Color jitter 0.8 + gray scale 0.2
- Gaussian blur (only on ImageNet, not CIFAR10)
Insights:
All of orientation, size, distribution and minor jitter of color should not affect the representation of an image.
Previous work:
Local DIM, CPC v2

None-linear projection head
- Dimensionality of projection head doesn't matter much.

LARS as optimizer
Stabilize the training process with large learning rate and batch size. 

Hyperparameters


'''