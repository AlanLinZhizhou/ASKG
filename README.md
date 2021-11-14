# ASKG
code for ASKG: Learning Sentiment-EnhancedWord Representation with External Knowledge Obtained by Weak Supervision

## Model Architecture
<p align="center">
    <img src="model.jpg" height="600"/>
</p> 

## Requirements
### Environment
```
* Python == 3.6.9
* Pytorch == 1.9.0
* CUDA == 10.2.89
* NVIDIA Tesla V100
* HuggingFaces Pytorch (also known as pytorch-pretrained-bert & transformers)
* Stanford CoreNLP (stanford-corenlp-full-2018-10-05)
* Numpy, Pickle, Tqdm, Scipy, etc. (See requirements.txt)
```

### Datasets
Datasets include:
```
* SST-5 
* SST-3 
* MR
* ALM
* AMAN
```
*All datasets are available on request to the owner of dataset.* 

### File Architecture (Selected important files)
```
-- /autosenti/generate_kgs_xxxx.py                                  ---> generate kgs wihtout external knowledge
-- /autosenti/add_externsenti2kgs.py                                ---> add external knowledge to kgs (i.e. sentiwordnet 3.0.0)
-- /autosenti/knowledgegraph.py                                     ---> knowledge filters and knowledge incorporation
-- /lexicon/senti_score.txt                                         ---> the converted results of sentiwordnet 3.0.0
-- /autosenti/kgs/xxx.spo                                           ---> automatically generated knwoledge
-- /datasets                                                        ---> datasets
-- /models                                                          ---> config files of pre-trained models
-- /skgframework                                                    ---> the framework of ASKG
```

## Get Started
* Download the ``bert-base.bin`` from [here](https://share.weiyun.com/EY7aJitJ), and save it to the ``models/`` directory.
* Download the ``bert-PT.bin`` from [here](https://share.weiyun.com/2fUr7Mgu), and save it to the ``models/`` directory.
* Download the ``GoogleNews-vectors-negative300.bin`` from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g), and save it to the ``autosenti/models/`` directory.
* Download the ``stanford-corenlp-full-2018-10-05.zip`` from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip), and unzip it to the ``autosenti/`` directory.

### Generate Sentiment Knowledge
a) You can directly download the sentiment knowledge from [here](https://share.weiyun.com/GVupcuPO) and save it to ``autosenti/`` directory.
b) Or you can also generate it on your own by utilizing our scripts








