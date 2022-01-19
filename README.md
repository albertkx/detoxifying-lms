# Detoxifying Language Models Risks Marginalizing Minority Voices

This repository contains the official code for our paper appearing in NAACL 2021.

Read our [paper](https://arxiv.org/abs/2104.06390) for more information about the experimental setup.

## Dependencies

The experiments depend on [Pytorch](https://pytorch.org/) and [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers).
We use the official code of respective papers to replicate their results (e.g., [GeDi](https://github.com/salesforce/GeDi) and [PPLM](https://github.com/uber-research/PPLM)]). 

## Setup

Create a new Anaconda environment and run the following:

```
./setup.sh
```

This will clone the PPLM and GeDi submodules, and install their dependencies.

As PPLM and GeDi require different HuggingFace Transformers versions, this script will also install both version 2.8 and version 3.4 as different pip packages.

Then, add your Perspecitve API key to scripts/score_generations.py if you need to score data/generations.

## Getting Started

Each of the controllable generation methods are placed in separate submodule/folders. Specifics of note:
+ `FT` contains all of the code for pretraining and DAPT finetuning. 
+ `transformers2` is a clone of Transformers 2.8 which is a GeDi dependency.

Examples of how to run training, generation, and evaluation for all the methods are available in the `Makefile`. Each of these commands references scripts in the `scripts/` folder. 

`scripts/` is organized as follows:
+ `scripts/data-processing` contains the scripts used to generate and/or filter training/evaluation data.
+ `scripts/generation` conatins the scripts used to perform both prompted and unprompted generation with each of the controllable generation methods.
+ `scripts/ppl` contains the scripts used for automated evaluation of model toxicity (perplexity)
+ `scripts/train` contains the scripts used to train all of the controllable generation methods.

`score_generations.py` can be flexibly used on any `.txt` file with the Perspective API and automatically resumes scoring if an error occurs.

## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Xu2021Detoxifying,
      Title = {Detoxifying Language Models Risks Marginalizing Minority Voices}, 
      Author = {Albert Xu and Eshaan Pathak and Eric Wallace and Suchin Gururangan and Maarten Sap and Dan Klein},
      Booktitle = {North American Chapter of the Association for Computational Linguistics}
      year={2021}
}
```

## Contributions and Contact

This code was developed by Albert Xu, Eric Wallace, and Eshaan Pathak. Contact us at albertxu3@berkeley.edu, ericwallace@berkeley.edu and eshaanpathak@berkeley.edu, respectively.
