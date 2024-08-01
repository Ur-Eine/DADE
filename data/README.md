
# Prerequisites
*   Python == 3.8, numpy == 1.20.3, faiss, tqdm, matplotlib

    ```shell
    conda create -n DADE python=3.8
    conda activate DADE
    pip install numpy==1.20.3
    conda install -c pytorch faiss-gpu
    pip install tqdm
    pip install matplotlib
    ```

# Datasets

The tested datasets are available at https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html. The datasets we tested in our paper include 'deep1M' 'gist' 'glove2.2' 'msong' 'tiny5m' 'word2vec', you can test some of them. For example:

1. Download the dataset GIST from ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz. The dataset is large. This step may take several minutes. The data format can be found in http://corpus-texmex.irisa.fr/.
    
    ```shell
    wget -O ./gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz --no-check-certificate
    ```

2. Unzip the dataset. 

    ```shell
    tar -zxvf ./gist.tar.gz -C ./
    ```

3. Some datasets lack ground truth, make sure the ground truth already exists before running subsequent code. For example, you can find 'gist_groundtruth.ivecs' in './gist/' . If it does not exist, create it.
    
    ```shell
    python getGT.py
    ```

4. Preprocess the dataset with random orthogonal transformation. 

    ```shell
    python randomized.py
    ```

5. Preprocess the dataset with PCA orthogonal transformation. 

    ```shell
    python pca.py
    python getEpsilon.py
    ```

6. Generate the clustering of the dataset for IVF. 

    ```shell
    python ivf.py
    ```