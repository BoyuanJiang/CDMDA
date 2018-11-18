# Unsupervised domain adaptation with target reconstruction and label confusion in the common subspace

This repository contains experiments code of our paper [**Unsupervised domain adaptation with target reconstruction and label confusion in the common subspace**](http://link.springer.com/article/10.1007/s00521-018-3846-x).

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

# Citation
```
@Article{Jiang2018,
         author="Jiang, Boyuan
         and Chen, Chao
         and Jin, Xinyu",
         title="Unsupervised domain adaptation with target reconstruction and label confusion in the common subspace",
         journal="Neural Computing and Applications",
         year="2018",
         month="Nov",
         day="15",
         issn="1433-3058",
         doi="10.1007/s00521-018-3846-x",
         url="https://doi.org/10.1007/s00521-018-3846-x"
}
```
