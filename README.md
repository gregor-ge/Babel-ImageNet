# Babel-ImageNet
This is the repository for our benchmark [Babel-ImageNet: Massively Multilingual Evaluation of Vision-and-Language Representations](https://arxiv.org/abs/2306.08658).

## About
Vision-and-language (VL) models with separate encoders for each modality (e.g., CLIP) have become the go-to models for zero-shot image classification and image-text retrieval. 
The bulk of the evaluation of these models is, however, performed with English text only: the costly creation of language-specific image-caption datasets has limited multilingual VL benchmarks to a handful of high-resource languages.  

We introduce Babel-ImageNet, a massively multilingual benchmark that offers (partial) translations of 1000 ImageNet labels to over 100 languages, built without resorting to machine translation (MT) or requiring manual annotation. 
We instead automatically obtain reliable translations of ImageNet concepts by linking them -- via shared WordNet synsets -- to BabelNet, a massively multilingual lexico-semantic network.

We evaluate13 different publicly available multilingual CLIP models on zero-shot image classification (ZS-IC) for each of the 100 Babel-ImageNet languages chosen for analysis, demonstrating a significant gap between English ImageNet performance and that of high-resource languages (e.g., German or Chinese), and an even bigger gap for low-resource languages (e.g., Sinhala or Lao). 
Crucially, we show that the models' ZS-IC performance on Babel-ImageNet highly correlates with their performance in image-text retrieval, validating that Babel-ImageNet is suitable for estimating the quality of the multilingual VL representation spaces for the vast majority of languages that lack gold image-text data.  


## Results
We benchmarked (pretty much) all public multilingual CLIP models on Babel-ImageNet and 
on three multilingual image-text retrieval datasets. 
Raw results are [here](results).

We prepared a [notebook](evaluation_scripts/results_analysis.ipynb) for easy browsing and analysis of the results.



## Usage

### Setup

We list the packages needed in [requirements.txt](requirements.txt). 
Both newer and older version *probably* work but use the specified version on problems.


### Data Preparation
We release the Babel-ImageNet labels [here](data/babel_imagenet.json). The JSON is a dictionary mapping each ISO language code to a tuple with 1) the indices of classes as they appear in ImageNet-1k and 2) the class label names.

We also release the prompts translated with NLLB-1.3b-distilled [here](data/nllb_dist13b_prompts.json).

Babel-ImageNet includes *only the labels* for the ImageNet classes - you need to download the images yourself.

Labels and prompts can be used in your code as (nearly) drop-in replacement for standard ImageNet zero-shot evaluation with OpenAI's labels and prompts. You only need to take care to process the images of the language subset of classes - see the class [BabelImageNet](eval_scripts/data.py) for an example on how to do this using the torchvision ImageNet dataset.


#### Retrieval
We offer the option to evaluate models for retrieval for XTD, XM3600, and xFlickrCo but you need to download the images (MSCOCO, Flickr30k, XM3600) yourself.


### Evaluation

[run_eval.py](run_eval.py) and [run_retrieval.py](run_retrieval.py) are simple CLI tools to evaluate your model:

#### Babel-ImageNet


```shell
python run_eval.py --imagenet_folder=$imagenet_folder --prompts="label,nllb_dist13b_prompts" --languages="298" \
  --num_workers=4 --batch_size=512 \
  --source="openclip" --from_pretrained="xlm-roberta-base-ViT-B-32@laion5b_s13b_b90k" \
  --out_file="results/babel-imagenet/openclip-xlmrb-vitb32"
```

#### Retrieval
```shell
python run_retrieval.py --image_folder=$image_folder --dataset_file="./data/xm3600.json"\
  --num_workers=4 --batch_size=512 \
  --source="openclip" --from_pretrained="xlm-roberta-base-ViT-B-32@laion5b_s13b_b90k" \
  --out_file="results/retrieval/xm3600/openclip-xlmrb-vitb32"
```

In [evaluation_scripts](evaluation_scripts), we have scripts to replicate evaluation for all models we tested.

#### New Models
Our code is easy to extend for new models:

* HuggingFace and open_clip models are supported out of the box with `--source="huggingface"|"openclip"` and
`--from_pretrained="$huggingface_model"|"$openclip_model@$pretrained"`
* Other models have to implement the [`CLIP` interface](benchmark/models/__init__.py) and add themselve to `get_model()` [here](benchmark/models/__init__.py).



### Reproducing the data
**Labels**  
Our data creation script can be found [here](data_scripts/dataset_creation_rpc.py).
We use the RPC mode of BabelNet, see [here](https://pypi.org/project/babelnet/) for more details on how to request
the data and set up the environment.

If you want to create labels for additional languages, simply adapt the language list used in the script.


**Prompts**   
For machine-translated prompts, see [this script](data_scripts/prompt_translation.py) on how the prompts were translated.


### Training
We have currently no plans to release training code because we use an internal, not-yet-released framework.
However, we are happy to help if you have questions about implementation details - simply open an issue.

## License
Babel-ImageNet is a processed version of BabelNet v5.2 downloaded from https://babelnet.org, made available with the BabelNet Non-Commercial License (see https://babelnet.org/full-license)

Our code is licensed under the MIT license.



## Citation
If you find this benchmark helpful, please cite the following publication:

```
@article{geigle2023babelimagenet,
  author       = {Gregor Geigle and
                  Radu Timofte and
                  Goran Glava\v{s}},
  title        = {{B}abel-{I}mage{N}et: Massively Multilingual Evaluation of Vision-and-Language Representations},
  journal      = {arXiv},
  volume       = {abs/2306.08658},
  year         = {2023},
  url          = {https://arxiv.org/abs/2306.08658},
  eprinttype    = {arXiv},
  eprint       = {2306.08658},
}
```

Also consider citing the following:

```
@inproceedings{babelnet,
  author    = {Roberto Navigli and
               Simone Paolo Ponzetto},
  editor    = {Jan Hajic and
               Sandra Carberry and
               Stephen Clark},
  title     = {BabelNet: Building a Very Large Multilingual Semantic Network},
  booktitle = {{ACL} 2010, Proceedings of the 48th Annual Meeting of the Association
               for Computational Linguistics, July 11-16, 2010, Uppsala, Sweden},
  pages     = {216--225},
  publisher = {The Association for Computer Linguistics},
  year      = {2010},
  url       = {https://aclanthology.org/P10-1023/},
  timestamp = {Fri, 06 Aug 2021 00:41:04 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/NavigliP10.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@inproceedings{imagenet,
  author    = {Jia Deng and
               Wei Dong and
               Richard Socher and
               Li{-}Jia Li and
               Kai Li and
               Li Fei{-}Fei},
  title     = {ImageNet: {A} large-scale hierarchical image database},
  booktitle = {2009 {IEEE} Computer Society Conference on Computer Vision and Pattern
               Recognition {(CVPR} 2009), 20-25 June 2009, Miami, Florida, {USA}},
  pages     = {248--255},
  publisher = {{IEEE} Computer Society},
  year      = {2009},
  url       = {https://doi.org/10.1109/CVPR.2009.5206848},
  doi       = {10.1109/CVPR.2009.5206848},
  timestamp = {Wed, 15 Sep 2021 14:13:01 +0200},
  biburl    = {https://dblp.org/rec/conf/cvpr/DengDSLL009.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```