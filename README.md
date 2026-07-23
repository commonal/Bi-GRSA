# Bi-GRSA

## Overview

This repository provides the official PyTorch implementation of the paper:

**Mitigating Popularity Bias in Graph Collaborative Filtering via Bilateral Semantic Alignment and Group-Balanced Ranking**

The current entry script `BISNA-PGR.py` implements the proposed Bi-GRSA framework.

## Requirements

- Python 3.9.7
- PyTorch 1.12.0+cu113
- NumPy 1.20.0
- Numba 0.54.1
- FAISS-GPU 1.7.2
- Pandas 1.3.4
- tqdm

## Data

Please organize the datasets as follows:

```text
Data/
├── gowalla/
│   ├── train.txt
│   └── test.txt
├── yelp2018/
│   ├── train.txt
│   └── test.txt
└── amazon-book/
    ├── train.txt
    └── test.txt
```

## Usage
### Yelp2018
```python
python BISNA-PGR.py --dataset_name yelp2018 --dataset_path Data --device 0 --layers_list "[5]" --cl_rate_list "[10]" --align_reg_list "[10]"
```
The GPU index can be changed using the --device argument.