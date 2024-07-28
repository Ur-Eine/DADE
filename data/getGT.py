import os
import numpy as np
import struct
import faiss

source = './'
datasets = ['deep1M', 'gist', 'glove2.2m', 'msong', 'tiny5m', 'word2vec']

def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def to_ivecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', x)
                fp.write(a)

if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        query_path = os.path.join(path, f'{dataset}_query.fvecs')
        gt_path = os.path.join(path, f'{dataset}_groundtruth.ivecs')

        # check if groundtruth exists
        if os.path.exists(gt_path):
            print(f'{dataset}\'s groundtruth already exists.')
            continue

        # read data and query vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        Q = read_fvecs(query_path)
        N, D = X.shape

        # build flat index
        print(f'Build flat index in {dataset}')
        flat_index = faiss.IndexFlatL2(D)
        flat_index.add(X)

        # search 100-nns by flat index and save groundtruth
        print(f'Search 100-nns in {dataset} by flat index and save groundtruth')
        _, GT = flat_index.search(Q, 100)
        GT = GT.astype(np.int32)
        to_ivecs(gt_path, GT)
