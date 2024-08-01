
import numpy as np
import faiss
import struct
import os

source = './'
datasets = ['deep1M', 'gist', 'glove2.2m', 'msong', 'tiny5m', 'word2vec']
# the number of clusters
K = 4096 

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

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

if __name__ == '__main__':

    for dataset in datasets:
        print(f"Clustering - {dataset}")
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
        randomzized_cluster_path = os.path.join(path, f"O{dataset}_centroid_{K}.fvecs")
        transformation_O_path = os.path.join(path, 'O.fvecs')
        pca_cluster_path = os.path.join(path, f"P{dataset}_centroid_{K}.fvecs")
        transformation_P_path = os.path.join(path, 'P.fvecs')

        # check if dataset exists
        if not os.path.exists(data_path):
            print(f'{dataset} does not exist.')
            continue

        # read data vectors
        X = read_fvecs(data_path)
        O = read_fvecs(transformation_O_path)
        P = read_fvecs(transformation_P_path)

        D = X.shape[1]
        
        # cluster data vectors
        index = faiss.index_factory(D, f"IVF{K},Flat")
        index.verbose = True
        index.train(X)
        centroids = index.quantizer.reconstruct_n(0, index.nlist)
        to_fvecs(centroids_path, centroids)

        # randomized centroids
        centroids_O = np.dot(centroids, O)
        to_fvecs(randomzized_cluster_path, centroids_O)

        # pca centroids
        centroids_P = np.dot(centroids, P)
        to_fvecs(pca_cluster_path, centroids_P)
