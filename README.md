# Class-Balanced Loss Based on Class Volume for Long-Tailed Object Recognition

This PyTorch code is based on https://github.com/abdullahjamal/Longtail_DA.

## Dependency

PyTorch (1.12.1)

scikit-learn (1.1.2)

h5py (3.7.0)

## Training

To train CIFAR10-LT with an imbalance factor of 200, run

```
python main.py --dataset cifar10 --num_classes 10 --imb_factor 0.005 --model sdv --w_norm E --w_epoch 160

```

The newly added parameters are:

```--model sdv```: Re-weight loss using class distribution volume.

```--w_norm E```: Normalize the re-weighting factors so that the expected weight is 1.

```--w_epoch 160```: Apply re-weighting after 160 epochs.
This is necessary because the predictions in early epochs are noisy.
