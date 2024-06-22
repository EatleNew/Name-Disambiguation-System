# NDS: Name Disambiguation System Based on GCCAD

**Source repo:** https://github.com/EatleNew/Name-Disambiguation-System/

This is the code implementation of the **team <u>Sumo</u>** in the competition **WhoIsWho-IND-KDD-2024**.

## Overview

Disambiguation of authors with the same name is crucial for ensuring academic fairness. This research develops a **N**ame **D**isambiguation **S**ystem based on the GCCAD (Graph Contrastive Coding for Anomaly Detection) model. We developed the original model with Sage convolution, batch normalization, residual connection, multi-head attention mechanism and AdamW optimizer, etc. The NDS model achieved impressive results, with an AUC of 0.766 on the test dataset and 0.716 on the validation dataset. **This result is the state-of-the-art(_sota_) among all the teams in our class who participated in this competition.**

## Prerequisites

### Pytorch environment

Create a virtual anaconda environment:

```bash
conda create -n your_env_name python=3.10.1
```

Active it and install the cuda version Pytorch:

```bash
conda install pytorch==2.1.0 torchvision torchaudio cudatoolkit -c pytorch
```

And then install the necessities:

```bash
pip install -r GCCAD/requirements.txt
```

### IND Dataset

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1_CX50fRxou4riEHzn5UYKg?pwd=gvza) with password gvza, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-WhoIsWho.zip) or [DropBox](https://www.dropbox.com/scl/fi/o8du146aafl3vrb87tm45/IND-WhoIsWho.zip?rlkey=cg6tbubqo532hb1ljaz70tlxe&dl=1).
Unzip the dataset and put files into `dataset/` directory.

## Running Steps

```
python encoding.py --path your_pub_dir --save_path embedding_save_path

python build_graph.py --author_dir train_author_path --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python build_graph.py --author_dir test_author_path --pub_dir your_pub_dir --save_dir save_path --embeddings_dir embedding_save_path

python train.py --train_dir train.pkl --test_dir valid.pkl
```

## Evaluation Results

| Method                 | AUC on validation set | AUC on test set |
| ---------------------- | --------------------- | --------------- |
| GCN                    | 0.58235               | -               |
| GCN+abstract           | 0.59781               | -               |
| GCCAD                  | 0.60865               | -               |
| GCCAD + abstract       | 0.62411               | -               |
| + SAGEConv + Batch     | 0.64617               | 0.72065         |
| + Residual + attention | 0.68923               | 0.74383         |
| + AdamW（=NDS）        | 0.71612               | 0.76573         |
