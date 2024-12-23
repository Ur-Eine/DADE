# Efficient Data-aware Distance Comparison Operation for High-Dimensinal Approximate Nearest Neighbor Search

## Introduction
This is the official implementation of the paper, entitled 'Efficient Data-aware Distance Comparison Operation for High-Dimensional Approximate Nearest Neighbor Search', which is accepted in VLDB 2025.

Most of existing algorithms of AKNNs can be decomposed into two components, i.e., candidate generation and distance comparison operations (DCOs), where DCOs is to check whether the distance between the query and the candidate greater than the given threshold (i.e., the distance of the K nearest neighbor to the query searched so far). In this study, we focus on speed up the process of DCOs.

The main idea of this paper is to approximate the distance between the query and the candidate in the space with lower dimension to save the cost of distance computation. To fulfill this goal, we propose a method called DADE, in which we theoretically prove that the distance estimation is unbiased in terms of the data distribution and optimized if the transformation is orthogonal. Moreover, we notice that the number of dimensions for distance estimation is different for different query-candidate pairs to have a sufficient confidence. Thus, we propose a hypothesis testing approach to adaptively determine the number of dimensions, where the probability in terms of the distance deviation is approximated from the data objects.

<!-- We note that we have included detailed comments of our core algorithms in 
* `./src/adsampling.h`
* `./src/hnswlib/hnswalg.h`
* `./src/ivf/ivf.h` -->

## Repo Description
We combine our DCO method with widely used AKNN search algorithms such as HNSW and IVF. The details of these algorithms can be found in `./src/hnswlib/hnswalg.h` and `./src/ivf/ivf.h`, respectively. The details of DADE can be found in `./src/dade.h`.

## Prerequisites

Following the seminal work for DCOs, i.e., ADSampling, we adopt Eigen in the index construction phase. Users can follow the below descriptions to enable Eigen in our repo.

* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `./src/`.

---
## GIST Reproduction

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. 

1. Download and preprocess the datasets, in which the data-aware orthogonal transformation is obtained though PCA(principal components analysis). Detailed instructions can be found in `./data/README.md`.
    ```sh
    cd ./data

    python randomized.py
    python pca.py
    python getEpsilon.py
    python ivf.py
    ```

2. Index the datasets. It could take several hours. 
    ```sh
    cd ../script

    # Index IVF/IVF+/IVF++/IVF*/IVF**
    sh index_ivf.sh

    # Index HNSW/HNSW+/HNSW++/HNSW*/HNSW**
    sh index_hnsw.sh

    # Index FLAT
    sh index_flat.sh
    ```
It should be noted that the speed-up techniques for DCOs is not applied in this phase since we want to eliminate the effect of different DCOs for the index structure. 

3. Test the queries of the datasets. The results are generated in `./results/`. Detailed configurations can be found in `./script/README.md`.
    ```sh
    # Search IVF/IVF+/IVF++/IVF*/IVF**
    sh search_ivf.sh

    # Search HNSW/HNSW+/HNSW++/IVF*/IVF**
    sh search_hnsw.sh

    # Search FLAT
    sh search_flat.sh
    ```

## Acknowledgement

This repo is mainly based on the seminal work for DCOs, called ADSampling(https://github.com/gaoj0017/ADSampling). If you think this repo is helpful, please consider to cite their paper.
