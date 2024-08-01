

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswlib.h>
#include <hnswlib/bruteforce.h>
#include <dade.h>

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;
const int MAXQ = 100;

long double rotation_time=0;

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t subk, BruteforceSearch<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.data_ + appr_alg.size_per_element_ * massQA[k * i + j], appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt){
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, BruteforceSearch<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive, char result_path[],FILE* original_stdout) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    dade::clear();

    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;  
        struct rusage run_start, run_end;
        GetCurTime( &run_start);
#endif
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k, adaptive);  
#ifndef WIN32
        GetCurTime( &run_end);
        GetTime( &run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;   
    }
    long double time_us_per_query = total_time / qsize + rotation_time;
    long double recall = 1.0f * correct / total;
    
    freopen(result_path, "a", stdout);
    if (dade::USE_PCA){
        cout << dade::significance << " " << recall * 100.0 << " " << time_us_per_query << " " << dade::tot_dimension + dade::tot_full_dist * vecdim << endl;
    }else if (dade::FIX_DIM){
        cout << dade::fixed_d << " " << recall * 100.0 << " " << time_us_per_query << " " << dade::tot_dimension + dade::tot_full_dist * vecdim << endl;
    }else{
        cout << dade::epsilon0 << " " << recall * 100.0 << " " << time_us_per_query << " " << dade::tot_dimension + dade::tot_full_dist * vecdim << endl;
    }stdout = original_stdout;
    freopen("/dev/tty", "w", stdout);
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, BruteforceSearch<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive, char epsilon_dir[], char result_path[], FILE* original_stdout) {
    if (dade::USE_PCA){
        vector<float> signs;
        char epsilon_path[256] = "";
        char significance_char[32] = "";
        for (int i=1; i<=12; ++i){
            signs.push_back(0.05*i);
        }
        for (float sign : signs){
            strcpy(epsilon_path, epsilon_dir);
            sprintf(significance_char, "%.*f", 2, sign);
            strcat(epsilon_path, significance_char);
            strcat(epsilon_path, ".fvecs");
            Matrix<float> E(epsilon_path);
            vector<float> temp;

            dade::significance = sign;
            temp.push_back(1.0e10);
            for(int i=0; i<vecdim; ++i){
                temp.push_back(E.data[0*E.d+i]);
            }
            swap(dade::epsilon, temp);
            test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive, result_path, original_stdout);
        }
    }else{
        if (dade::FIX_DIM){
            int N_exp = 32;
            int delta_d = vecdim / N_exp;
            vector<int> fds;
            for (int i=1; i<=N_exp; ++i){
                fds.push_back(i * delta_d);
            }
            for (int fd : fds){
                dade::fixed_d = fd;
                test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive, result_path, original_stdout);
            }
        }
        else{
            vector<float> eps;
            for (int i=1; i<=8; ++i){
                eps.push_back(i * 0.4 + 0.1);
            }
            for (float ep : eps) {
                dade::epsilon0 = ep;
                test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive, result_path, original_stdout);
            }
        }
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"k",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"gap",                         required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
        {"eigenvalue_path",             required_argument, 0, 'l'},
        {"epsilon_dir",                required_argument, 0, 's'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";
    char eigenvalue_path[256] = "";
    char epsilon_dir[256] = "";

    int randomize = 0;
    int subk=100;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:l:s:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg)dade::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)dade::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'l':
                if(optarg)strcpy(eigenvalue_path, optarg);
                break;
            case 's':
                if(optarg)strcpy(epsilon_dir, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
        }
    }

    
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    Matrix<float> P(transformation_path);
    Matrix<float> L(eigenvalue_path);

    FILE* original_stdout = stdout;
    if(randomize){
        StopW stopw = StopW();
        Q = mul(Q, P);
        rotation_time = stopw.getElapsedTimeMicro() / Q.n;
        dade::D = Q.d;
        if(randomize == 2){
            for(int i=0; i<Q.d; ++i){
                dade::lmds.push_back(L.data[0*L.d+i]);
            }dade::USE_PCA = true;
            dade::compute_cdf_lmd();
        }if(randomize == 3 || randomize == 4){
            dade::FIX_DIM = true;
            randomize = 3;
        }
    }
    
    L2Space l2space(Q.d);
    BruteforceSearch<float> *appr_alg = new BruteforceSearch<float>(&l2space, index_path);

    size_t k = G.d;
    size_t qsize = Q.n < MAXQ ? Q.n : MAXQ;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

    get_gt(G.data, Q.data, appr_alg->max_elements_, qsize, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, qsize, *appr_alg, Q.d, answers, subk, randomize, epsilon_dir, result_path, original_stdout);

    return 0;
}
