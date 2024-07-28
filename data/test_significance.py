import os
import numpy as np
import struct
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# base config
N_s = 100000
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

        # uniformly sample N_s pairs
        print(f"Uniformly sample {N_s} int pairs with range from 0 to {N-1}")
        pairs = np.random.randint(0, N, (2, N_s))
        XO0 = XO[pairs[0]]
        XO1 = XO[pairs[1]]
        XP0 = XP[pairs[0]]
        XP1 = XP[pairs[1]]

        # compute dis and dis' about d
        print(f"compute dis and dis' about d, d from 1 to {D}")
        DO_SQR = np.zeros((N_s, D))
        DP_SQR = np.zeros((N_s, D))
        sumO = np.zeros((N_s))
        sumP = np.zeros((N_s))
        for d in tqdm(range(D)):
            sumO += (XO0[:, d] - XO1[:, d]) * (XO0[:, d] - XO1[:, d])
            sumP += (XP0[:, d] - XP1[:, d]) * (XP0[:, d] - XP1[:, d])
            DO_SQR[:, d] = sumO * D / (d+1)
            DP_SQR[:, d] = sumP * CDF_LMD[D-1] / CDF_LMD[d]

        # test significance about d
        SO1 = np.zeros(D)
        SP1 = np.zeros(D)
        SO2 = np.zeros(D)
        SP2 = np.zeros(D)
        DT_SQR = DO_SQR[:, D-1]

        epsilon = 0.01
        flag = False
        while flag:
            print(f"test significance about d, d from 1 to {D}")
            ratio1 = (1 + epsilon) * (1 + epsilon)
            ratio2 = (1 - epsilon) * (1 - epsilon)
            for d in tqdm(range(D)):
                DO_SQR_d = DO_SQR[:, d]
                DP_SQR_d = DP_SQR[:, d]
                SO1[d] += len(np.where(DO_SQR_d > ratio1 * DT_SQR)[0])
                SP1[d] += len(np.where(DP_SQR_d > ratio1 * DT_SQR)[0])
                SO2[d] += len(np.where(DO_SQR_d < ratio2 * DT_SQR)[0])
                SP2[d] += len(np.where(DP_SQR_d < ratio2 * DT_SQR)[0])
            SO1 /= N_s
            SP1 /= N_s
            SO2 /= N_s
            SP2 /= N_s

            maxy = np.max(np.array([SO1, SP1, SO2, SP2]))
            plt.style.use(good_style[3])
            plt.figure()
            plt.title(dataset, font=FONT)
            plt.xticks(font=FONT)
            plt.xlim((0, D * 1.1))
            plt.xlabel('Dim', fontdict=FONT)
            plt.yticks(font=FONT)
            plt.ylim((0, maxy * 1.1))
            plt.ylabel('Significance', fontdict=FONT)
            plt.plot(SO1, color=common_color[0])
            plt.plot(SO2, color=common_color[1])
            plt.plot(SP1, color=common_color[2])
            plt.plot(SP2, color=common_color[3])
            plt.legend(('random s1', 'random s2', 'pca s1', 'pca s2'), prop=font, loc='upper right', handlelength=3, numpoints=1, markerscale=0.7)
            plt.savefig(f'./fig/{dataset}_Sign-dim_epsilon={epsilon}.png', bbox_inches='tight', dpi=1000)

            import pdb;pdb.set_trace()
    
        N_ss = 1000
        flag = False
        while flag:
            print(f"draw dis'2-dis about d, d from 1 to {D}")
            DO_SQR_DIF = (DO_SQR[:N_ss] - DT_SQR[:N_ss, np.newaxis]).reshape(-1)
            DP_SQR_DIF = (DP_SQR[:N_ss] - DT_SQR[:N_ss, np.newaxis]).reshape(-1)
            D_list = np.array([i for i in range(1, D+1)])
            draw_x = np.zeros((N_ss, D))
            draw_x = (draw_x + D_list[np.newaxis, :]).reshape(-1)

            maxy = np.max(np.array([DO_SQR_DIF, DP_SQR_DIF]))
            miny = np.min(np.array([DO_SQR_DIF, DP_SQR_DIF]))
            leny = maxy - miny
            plt.style.use(good_style[3])
            plt.figure()
            plt.title(dataset, font=FONT)
            plt.xticks(font=FONT)
            plt.xlim((0, D * 1.1))
            plt.xlabel('Dim', fontdict=FONT)
            plt.yticks(font=FONT)
            # plt.ylim((miny+0.2*leny, maxy-0.2*leny))
            plt.ylim((-4, 4))
            plt.ylabel('Dis\'2-Dis2', fontdict=FONT)

            s_=0.8
            alpha_=0.02
            import pdb;pdb.set_trace()
            plt.scatter(draw_x, DO_SQR_DIF, color=common_color[0], s=s_, alpha=alpha_, linewidths=0)
            plt.scatter(draw_x, DP_SQR_DIF, color=common_color[1], s=s_, alpha=alpha_, linewidths=0)

            plt.legend(('random', 'pca'), prop=font, loc='upper right', handlelength=3, numpoints=1, markerscale=0.7)
            plt.savefig(f'./fig/{dataset}_Diff-dim_scatter.png', bbox_inches='tight', dpi=3000)

            import pdb;pdb.set_trace()

        quantile = 0.8
        flag = False
        while flag:
            print(f"draw dis'2-dis2(quantile={quantile}) about d, d from 1 to {D}")
            DO_SQR_DIF = np.sort(DO_SQR - DT_SQR[:, np.newaxis], axis=0)
            DP_SQR_DIF = np.sort(DP_SQR - DT_SQR[:, np.newaxis], axis=0)
            LB_I = int(N_s * (1 - quantile))
            UB_I = int(N_s * quantile)
            LB_O = DO_SQR_DIF[LB_I,:]
            UB_O = DO_SQR_DIF[UB_I,:]
            LB_P = DP_SQR_DIF[LB_I,:]
            UB_P = DP_SQR_DIF[UB_I,:]
            draw_x = np.array([i for i in range(1, D+1)])

            maxy = np.max(np.array([LB_O, UB_O, LB_P, UB_P]))
            miny = np.min(np.array([LB_O, UB_O, LB_P, UB_P]))
            leny = maxy - miny
            plt.style.use(good_style[3])
            plt.figure()
            plt.title(dataset, font=FONT)
            plt.xticks(font=FONT)
            plt.xlim((0, D * 1.05))
            plt.xlabel('Dim', fontdict=FONT)
            plt.yticks(font=FONT)
            plt.ylim((miny-0.05*leny, maxy+0.05*leny))
            plt.ylabel(f'Diff(quantile={quantile})', fontdict=FONT)

            plt.plot(draw_x, LB_O, color=common_color[0])
            plt.plot(draw_x, LB_P, color=common_color[1])
            plt.plot(draw_x, UB_O, color=common_color[0])
            plt.plot(draw_x, UB_P, color=common_color[1])

            plt.legend(('random', 'pca'), prop=font, loc='upper right', handlelength=3, numpoints=1, markerscale=0.7)
            plt.savefig(f'./fig/{dataset}_Diff-dim_quantile={quantile}.png', bbox_inches='tight', dpi=1000)

            import pdb;pdb.set_trace()

        quantile = 0.9
        N_show = 64
        flag = True
        while flag:
            step = int(D / N_show)
            print(f"draw dis'2 / dis2 - 1(quantile={quantile}) about d, d from 1 to {D}, step={step}")
            DO_SQR_DIF = np.sort(DO_SQR / DT_SQR[:, np.newaxis] - 1, axis=0)
            DP_SQR_DIF = np.sort(DP_SQR / DT_SQR[:, np.newaxis] - 1, axis=0)
            LB_I = int(N_s * (1 - quantile))
            UB_I = int(N_s * quantile)
            LB_O = DO_SQR_DIF[LB_I,:][::step]
            UB_O = DO_SQR_DIF[UB_I,:][::step]
            LB_P = DP_SQR_DIF[LB_I,:][::step]
            UB_P = DP_SQR_DIF[UB_I,:][::step]
            draw_x = np.array([i for i in range(1, D+1)])[::step]

            maxy = np.max(np.array([LB_O, UB_O, LB_P, UB_P]))
            miny = np.min(np.array([LB_O, UB_O, LB_P, UB_P]))
            leny = maxy - miny
            plt.style.use(good_style[3])
            plt.figure()
            plt.title(dataset, font=FONT)
            plt.xticks(font=FONT)
            plt.xlim((0, D * 1.05))
            plt.xlabel('of Dimensions', fontdict=FONT)
            plt.yticks(font=FONT)
            plt.ylim((miny-0.05*leny, maxy+0.05*leny))
            plt.ylabel(f'Diff_ratio(quantile={quantile})', fontdict=FONT)

            plt.plot(draw_x, LB_O, color=common_color[0], linestyle=good_linestyle[1], 
                     linewidth=1, marker=good_marker[3], markeredgewidth=1.2, markersize=4)
            plt.plot(draw_x, LB_P, color=common_color[1], linestyle=good_linestyle[1], 
                     linewidth=1, marker=good_marker[4], markeredgewidth=1.2, markersize=4)
            plt.plot(draw_x, UB_O, color=common_color[0], linestyle=good_linestyle[1], 
                     linewidth=1, marker=good_marker[3], markeredgewidth=1.2, markersize=4)
            plt.plot(draw_x, UB_P, color=common_color[1], linestyle=good_linestyle[1], 
                     linewidth=1, marker=good_marker[4], markeredgewidth=1.2, markersize=4)

            plt.legend(('random', 'pca'), prop=font, loc='upper right', handlelength=3, numpoints=1, markerscale=0.7)
            plt.savefig(f'./fig/{dataset}_Diff_ratio-dim_quantile={quantile}.pdf', bbox_inches='tight', dpi=1000)

            import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()
