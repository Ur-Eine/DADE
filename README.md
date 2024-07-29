# Efficient Data-aware Distance Comparison Operation for High-Dimensinal Approximate Nearest Neighbor Search

## Introduction
This is the official implementation of the paper, entitled 'Efficient Data-aware Distance Comparison Operation for High-Dimensional Approximate Nearest Neighbor Search'.

Most of existing algorithms of AKNNs can be decomposed into two components, i.e., candidate generation and distance comparison operations (DCOs), where DCOs is to check whether the distance between the query and the candidate greater than the given threshold (i.e., the distance of the K nearest neighbor to the query searched so far). In this study, we focus on speed up the process of DCOs.

The main idea of this paper is to approximate the distance between the query and the candidate in the space with lower dimension to save the cost of distance computation. To fulfill this goal, we propose a method called DAD, in which we theoretically prove that the distance estimation is unbiased in terms of the data distribution and optimal if the transformation is orthogonal. Moreover, we notice that the number of dimensions for distance estimation is different for different query-candidate pairs to have a sufficient confidence. Thus, we propose a hypothesis testing approach to adaptively determine the number of dimensions, where the probability in terms of the distance deviation is approximated from the data objects.

<!-- We note that we have included detailed comments of our core algorithms in 
* `./src/adsampling.h`
* `./src/hnswlib/hnswalg.h`
* `./src/ivf/ivf.h` -->

## Repo Description
We combine our DCO method with widely used AKNN search algorithms such as HNSW and IVF. The details of these algorithms can be found in `./src/hnswlib/hnswalg.h` and `./src/ivf/ivf.h`, respectively. The details of DAD can be found in `./src/dad.h`.

## Prerequisites

Following the seminal work for DCOs, i.e., ADSampling, we adopt Eigen in the index construction phase. Users can follow the below descriptions to enable Eigen in our repo.

* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `./src/`.

---
## GIST Reproduction

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. 

1. Download and preprocess the datasets, in which the data-aware orthogonal transformation is obtained though PCA(principal components analysis). Detailed instructions can be found in `./data/README.md`.
   ```
    python ./data/pca.py
    ```

2. Index the datasets. It could take several hours. 
    ```sh
    # Index IVF/IVF+/IVF++/IVF*/IVF**
    ./script/index_ivf.sh

    # Index HNSW/HNSW+/HNSW++/HNSW*/HNSW**
    ./script/index_hnsw.sh
    ```
It should be noted that the speed-up techniques for DCOs is not applied in this phase since we want to eliminate the effect of different DCOs for the index structure. 

3. Test the queries of the datasets. The results are generated in `./results/`. Detailed configurations can be found in `./script/README.md`.
    ```sh
    # Index IVF/IVF+/IVF++/IVF*/IVF**
    ./script/search_ivf.sh

    # Index HNSW/HNSW+/HNSW++/IVF*/IVF**
    ./script/search_hnsw.sh
    ```

## Acknowledgement

This repo is mainly based on the seminal work for DCOs, called ADSampling(https://github.com/gaoj0017/ADSampling). If you think this repo is helpful, please consider to cite their paper.

```
@article{DBLP:journals/pacmmod/GaoL23,
  author       = {Jianyang Gao and
                  Cheng Long},
  title        = {High-Dimensional Approximate Nearest Neighbor Search: with Reliable
                  and Efficient Distance Comparison Operations},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {1},
  number       = {2},
  pages        = {137:1--137:27},
  year         = {2023},
  url          = {https://doi.org/10.1145/3589282},
  doi          = {10.1145/3589282},
  timestamp    = {Fri, 07 Jul 2023 23:32:33 +0200},
  biburl       = {https://dblp.org/rec/journals/pacmmod/GaoL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
