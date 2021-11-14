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
* Download the ``bert-base.bin`` from [here](https://share.weiyun.com/EY7aJitJ) or [here](https://drive.google.com/file/d/1Uq-fuDo8qPv2FywCnbxW83ymIo4Xo3BJ/view?usp=sharing), and save it to the ``models/`` directory.
* Download the ``bert-PT.bin`` from [here](https://share.weiyun.com/2fUr7Mgu) or [here](https://drive.google.com/file/d/1tXjpMLLR4wdYT7qe4IT6U9Wh5fis09hY/view?usp=sharing), and save it to the ``models/`` directory.
* Download the ``GoogleNews-vectors-negative300.bin.gz`` from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g), and save it to the ``autosenti/models/`` directory.
* Download the ``stanford-corenlp-full-2018-10-05.zip`` from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip), and unzip it to the ``autosenti/`` directory.

#### Generate Sentiment Knowledge
* a) You can directly download the sentiment knowledge from [here](https://share.weiyun.com/GVupcuPO) or [here](https://drive.google.com/drive/folders/1YF1l-9ccUgTIT0j5Psmk6x7G7bSMIxok?usp=sharingand) save it to ``autosenti/`` or ``autosenti/kg`` directory.
* b) Or you can also generate it on your own by utilizing our scripts
```
The process of generating sentiment knowlegde on your own
(1).Generate sentiment knowledge without external knowledge
run generate_kgs_xxx.py
(2).Add external knowledge to sentiment knowledge
python add_externsenti2kgs.py \
      --input_spo_path ./kgs/xx.spo \
      --output_spo_path ./kgs/xx 
(3).Add your generated spo file to ``autosenti/config.py``
```

## Sentiment Classification and Emotion Detection
### Classification example

Run example on sst5 with SKG-BERT:
```sh
CUDA_VISIBLE_DEVICES=0 nohup python3 -u run_classifier.py \
    --pretrained_model_path ./models/bert-base.bin \
    --vocab_path  ./models/google_uncased_en_vocab.txt \
    --train_path  ./datasets/sst5/train.tsv \
    --dev_path  ./datasets/sst5/dev.tsv \
    --test_path  ./datasets/sst5/test.tsv \
    --output_model_path  ./models/sst5/modelsst5-bertbase.bin \
    --config_path ./models/bert/base_config.json \
    --epochs_num 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder transformer \
    --mask fully_visible \
    --kg_name sst5_addsenti \
    --workers_num 1 \
    --em_weight 0.6 \
    --mylambda 0.6 \
    --k0 0 \
    --k 2 \
    --l_ra0 1 \
    --l_ra 11 \
    --step 0.01 \
    --report_steps 20 \
```

Run example on sst5 with SKG-BERT-PT:
```sh
CUDA_VISIBLE_DEVICES=0 nohup python3 -u run_classifier.py \
    --pretrained_model_path ./models/bert-PT.bin \
    --vocab_path  ./models/google_uncased_en_vocab.txt \
    --train_path  ./datasets/sst5/train.tsv \
    --dev_path  ./datasets/sst5/dev.tsv \
    --test_path  ./datasets/sst5/test.tsv \
    --output_model_path  ./models/sst5/modelsst5-bertPT.bin \
    --config_path ./models/bert/base_config.json \
    --epochs_num 5 \
    --batch_size 32 \
    --embedding word_pos_seg \
    --encoder transformer \
    --mask fully_visible \
    --kg_name sst5_addsenti \
    --workers_num 1 \
    --em_weight 0.6 \
    --mylambda 0.6 \
    --k0 0 \
    --k 2 \
    --l_ra0 1 \
    --l_ra 11 \
    --step 0.01 \
    --report_steps 20 \
```







