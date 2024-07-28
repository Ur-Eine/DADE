import os
import numpy as np
import struct
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# base config
N_s = 100000
significances = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
source = './'
datasets = ['gist', 'deep1M', 'glove2.2m', 'msong', 'tiny5m', 'word2vec']

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

def PCA(x):
    x = copy.deepcopy(np.mat(x))
    x -= x.mean(axis=1)
    lmd, w = np.linalg.eig(x * x.T)
    return np.array(w.astype('float32')), np.array(lmd.astype('float32'))[np.newaxis, :]

if __name__ == "__main__":
    
    for dataset in datasets:
        np.random.seed(0)
        
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        transformed_path = os.path.join(path, 'P.fvecs')
        lambda_path = os.path.join(path, 'LMD.fvecs')
        significance_dir = os.path.join(path, 'E/')
        os.makedirs(significance_dir, exist_ok=True)

        # read data vectors & projection matrix & eigenvalues
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        P = read_fvecs(transformed_path)
        LMD = read_fvecs(lambda_path)
        N, D = X.shape

        # project & compute cdf of eigenvalues
        print(f"project {dataset} & compute cdf of eigenvalues.")
        XP = np.dot(X, P)
        CDF_LMD = np.cumsum(LMD)

        # uniformly sample N_s pairs
        print(f"Uniformly sample {N_s} int pairs with range from 0 to {N-1}")
        pairs = np.random.randint(0, N, (2, N_s))
        XP0 = XP[pairs[0]]
        XP1 = XP[pairs[1]]

        # compute dis and dis' about d
        print(f"compute dis and dis' about d, d from 1 to {D}")
        DP_SQR = np.zeros((N_s, D))
        sumP = np.zeros((N_s))
        for d in tqdm(range(D)):
            sumP += (XP0[:, d] - XP1[:, d]) * (XP0[:, d] - XP1[:, d])
            DP_SQR[:, d] = sumP * CDF_LMD[D-1] / CDF_LMD[d]

        # compute & save epsilons about d
        for significance in significances:
            print(f"compute epsilons (i.e. dis'2 / dis2 - 1) when significance={significance} about d, d from 1 to {D}")
            DT_SQR = DP_SQR[:, D-1]
            DP_SQR_DIF = np.sort(DP_SQR / DT_SQR[:, np.newaxis] - 1, axis=0)[::-1]
            EP = DP_SQR_DIF[int(N_s * significance),:][np.newaxis, :]
            epsilon_path = os.path.join(significance_dir, "{:.2f}".format(significance) + '.fvecs')
            to_fvecs(epsilon_path, EP)
