#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/bruteforce.h"

using namespace std;
using namespace hnswlib;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:m:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'i':
                if(optarg){
                    strcpy(index_path, optarg);
                }
                break;
        }
    }
    
    Matrix<float> *X = new Matrix<float>(data_path);
    size_t D = X->d;
    size_t N = X->n;
    size_t report = 50000;

    L2Space l2space(D);
    BruteforceSearch<float>* appr_alg = new BruteforceSearch<float> (&l2space, N);

    for(int i=0;i<N;i++){
        appr_alg->addPoint(X->data + i * D, i);
        if(i % report == 0){
            cerr << "Processing - " << i << " / " << N << endl;
        }
    }

    appr_alg->saveIndex(index_path);
    return 0;
}
