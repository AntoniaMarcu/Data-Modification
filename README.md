<div align="center">
  
# On the Effects of Artificial Data Modification

This is the __official__ code for the paper ["On the Effects of Artificial Data Modification"](https://proceedings.mlr.press/v162/marcu22a.html).

</div>

## About

Data distortion is commonly applied in vision models during both training (e.g methods like MixUp and CutMix) and during evaluation (e.g. for measuring shape-texture bias and robustness). 
Data modification can introduce artificial information. It is often assumed that the resulting artefacts are detrimental to training, whilst being negligible when analysing models. The code in this repository was used to investigate these assumptions. In our [paper](https://proceedings.mlr.press/v162/marcu22a.html) we conclude that in some cases they are unfounded and lead to incorrect results.

## Experiments

### Data Interference (DI) index

To compute the DI index, we need a few different runs of the same model. 
In our paper we trained the model 5 times. 
The script assumes that the models were saved under the same base name, followed by the index of the run. 
So the format is of the type ```f'base-path_{run_id}'```.

```
usage: DI.py [--distortion {cutOut_restricted,cutOut_unrestricted,cutMix_restricted,cutMix_unrestricted,shuffle}]
             [--device DEVICE] [--model MODEL] [--model-path MODEL_PATH]
             [--dataset {cifar10,cifar100,fashion,imagenet}]
             [--dataset-path DATASET_PATH] [--augment AUGMENT]
             [--batch-size BATCH_SIZE] [--n-runs N_RUNS]
```

Say we want to compute the DI index for preAct-ResNet18 models trained on CIFAR-10 when distorting with black patches.
Assuming the models have paths f'basic_{id}.pt', where id is in [0,4], we'd run the following:

```python DI.py --dataset=cifa10 --dataset-path='./data/cifar10' --model=resnet18 --model-path='basic_' --distortion='cutOut_restricted''```

As underlined in the paper, the DI index is designed to compare the Data Interference of different models. 

### iOcclusion as a fairer alternative

```
usage: occlusion.py [--measurement {iOcclusion_gradcam,cutOcclusion,iOcclusion_random,none}]
                     [--dataset {cifar10,cifar100,fashion,imagenet}]
                     [--augment AUGMENT] [--device DEVICE]
                     [--proportion PROPORTION] [--model-path MODEL_PATH]
                     [--dataset-path DATASET_PATH]
                     [--dataset-proportion DATASET_PROPORTION]
```

Note that for masking with iOcclusion we provide 2 variants: 
* random masking (noisier, but less computationally intensive): iOcclusion_random
* saliency-based masking: iOcclusion_gradcam

To compute iOcclusion for a preAct-ResNet18 model trained on CIFAR-10 we'd run:

```python iOcclusion.py --measurement=iOcclusion_gradcam --datset=cifar10 --dataset-path='./data/cifar10' --model-path='basic_0.pt'```

## Citation

If you find this repository useful, please cite the paper:

      @InProceedings{pmlr-v162-marcu22a,
  		  title = 	 {On the Effects of Artificial Data Modification},
		  author =       {Marcu, Antonia and Prugel-Bennett, Adam},
		  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
		  pages = 	 {15050--15069},
		  year = 	 {2022},
		  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
		  volume = 	 {162},
		  series = 	 {Proceedings of Machine Learning Research},
		  month = 	 {17--23 Jul},
		  publisher =    {PMLR},
  		  pdf = 	 {https://proceedings.mlr.press/v162/marcu22a/marcu22a.pdf},
  		  url = 	 {https://proceedings.mlr.press/v162/marcu22a.html},

