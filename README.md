# Babel-ImageNet
This is the repository for our benchmark [TITLE](url).

## Usage


### Using the benchmark
We release the Babel-ImageNet labels [here](data/babel_imagenet.json). We also release the prompts translated with NLLB-1.3b-distilled [here](data/nllb_dist13b_prompts.json).

Babel-ImageNet includes *only the labels* for the ImageNet classes - you need to download the images yourself.

Labels and prompts can be used in your code as (nearly) drop-in replacement for standard ImageNet zero-shot evaluation with OpenAI's labels and prompts. You only need to take care to process the images of the language subset of classes - see the class [BabelImageNet](eval_scripts/data.py) for an example on how to do this using the torchvision ImageNet dataset.

### Evaluation code
[eval.py](eval_scripts/eval.py) is an efficient* script to evaluate Babel-ImageNet on many languages at once (by caching image embeddings).
The models used in our paper are ready for evaluation and adding your own model is quite straightforward.

*Evaluating OpenCLIP ViT-B32 on all languages with labels, English prompts and MT prompts on a RTX 3090 takes <40min (~5min for image encoding and then a few second per language+prompt).

See [requirements.txt](requirements.txt) for necessary packages. AltCLIP has no corresponding package, so get the code [here](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/hf_altclip).

```shell
python eval_scripts/eval.py \
--imagenet_folder /PATH/TO/IMAGENET/val
```

List of arguments:
```shell
python eval_scripts/eval.py --help
```

### Reproduce the data
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
TODO
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