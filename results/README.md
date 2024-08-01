# Results
## IVF or HNSW
The results of the experiments of ivf or hnsw are stored in the files with their names formatted in `./{dataset}_{algorithm}_{Some parameters}_{0/1/2/3/4}.log`.
*   0 - AKNN
*   1 - AKNN++
*   2 - AKNN+
*   **3 - AKNN\*\***
*   4 - AKNN\*

In the files, each line contains 4 numbers. They represent 
1. search parameter ($N_{ef}$ for HNSW and $N_{probe}$ for IVF)
2. recall (%)
3. average running time per query (us)
4. total evaluated dimensionality

## Linear Scan
The results of the experiments of linear scan are stored in the files with their names formatted in `./{dataset}_FLAT_{Some parameters}_{0/1/2/3/4}.log`.
*   0 - Linear Scan
*   1 - Linear Scan with ADSampling
*   2 - Linear Scan with DADE
*   3 - Linear Scan with Random Orthogonal Transformation (Fixed Dimension)
*   4 - Linear Scan with PCA Orthogonal Transformation (Fixed Dimension)

In the files, each line contains 4 numbers. They represent 
1. search parameter (Meaningless numbers for 0, $\epsilon_0$ for 1, significance for 2 and Fixed Dimension for 3,4)
2. recall (%)
3. average running time per query (us)
4. total evaluated dimensionality
