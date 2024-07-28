import os
import numpy as np
import struct
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# base config
source = './'
datasets = ['deep1M', 'gist']
# datasets = ['deep1M', 'gist', 'glove2.2m', 'msong', 'tiny5m', 'word2vec']

# draw config
FONT = {'family': 'Times New Roman', 'size': 18}
font = {'family': 'Times New Roman', 'size': 14}
good_style = ['classic', 'ggplot', 'grayscale', 'seaborn-v0_8']
good_marker = ['^', 's', 'p', 'o', 'x']
good_linestyle = ['--', (0, (5, 2)), '-.', '-']
common_color = ['r', 'c', 'g', 'b']

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

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

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
        os.makedirs(f'./fig/', exist_ok=True)

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        N, D = X.shape

        # generate random orthogonal matrix, store it and apply it
        print(f"Randomizing {dataset} of dimensionality {D}.")
        O = Orthogonal(D)
        XO = np.dot(X, O)

        # generate pca orthogonal matrix & eigenvalues, store it and apply it
        print(f"Randomizing {dataset} of dimensionality {D} by PCA.")
        P, LMD = PCA(X.T)
        XP = np.dot(X, P)
        CDF_LMD = np.cumsum(LMD)

        # compute variance about d
        print(f"compute variance about d, d from 1 to {D}")
        VAR_O = np.var(XO, axis=0)
        VAR_P = np.var(XP, axis=0)

        flag = True
        N_show = 64
        while flag:
            step = int(D / N_show)
            print(f"draw variance about d, d from 1 to {D}, step={step}")
            VAR_O_SHOW = VAR_O[::step]
            VAR_P_SHOW = VAR_P[::step]
            draw_x = np.array([i for i in range(1, D+1)])[::step]

            maxy = np.max(np.array([VAR_O_SHOW, VAR_P_SHOW]))
            miny = np.min(np.array([VAR_O_SHOW, VAR_P_SHOW]))
            leny = maxy - miny
            plt.style.use(good_style[3])
            plt.figure()
            plt.title(dataset, font=FONT)
            plt.xticks(font=FONT)
            plt.xlim((0, D * 1.05))
            plt.xlabel('of Dimensions', fontdict=FONT)
            plt.yticks(font=FONT)
            plt.ylim((miny-0.02*leny, maxy+0.05*leny))
            plt.ylabel(f'Variance', fontdict=FONT)

            plt.plot(draw_x, VAR_O_SHOW, color=common_color[0], linestyle=good_linestyle[0], 
                     linewidth=1, marker=good_marker[3], markeredgewidth=1.2, markersize=4)
            plt.plot(draw_x, VAR_P_SHOW, color=common_color[1], linestyle=good_linestyle[1], 
                     linewidth=1, marker=good_marker[4], markeredgewidth=1.2, markersize=4)

            plt.legend(('random', 'pca'), prop=font, loc='upper right', handlelength=3, numpoints=1, markerscale=0.7)
            plt.savefig(f'./fig/{dataset}_Variance.pdf', bbox_inches='tight', dpi=1000)

            flag = False
            # import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()
