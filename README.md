# Unsupervised domain adaptation with target reconstruction and label confusion in the common subspace

This repository contains experiments code of our paper **Unsupervised domain adaptation with target reconstruction and label confusion in the common subspace**.

# How to run
First download the dataset from [MEGA](https://mega.nz/#!4eRWVCKL!sMuftfE6cRkZZdePrFoGynpevRUnpYT1MRwT0gQpx3s), then extract files to *data* folder.

Train the model on SVHN->MNIST scenario

```bash
python main.py --mode 'train' --method 'recon' --source 'svhn' --target 'mnist' 
```

Test the model on testset

```bash
python main.py --mode 'test' --method 'recon' --source 'svhn' --target 'mnist' --device '/cpu:0'
```

# Thanks 
This code based on this [repository](https://github.com/pmorerio/minimal-entropy-correlation-alignment), thanks for the authors.

