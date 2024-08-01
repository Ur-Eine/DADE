

cd ..

g++ ./src/search_hnsw.cpp -O3 -o ./src/search_hnsw -I ./src/

path=./data/
result_path=./results/

ef=500
M=16
K=100
sign=0.10
gap=32

for data in 'deep1M' 'gist' 'glove2.2m' 'msong' 'tiny5m' 'word2vec'
do
    for randomize in $(seq 0 4)
    do
        if [ "$randomize" -eq 1 ]
        then 
            echo "HNSW++"
            index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
            trans="${path}/${data}/O.fvecs"
        elif [ "$randomize" -eq 2 ]
        then 
            echo "HNSW+"
            index="${path}/${data}/O${data}_ef${ef}_M${M}.index"
            trans="${path}/${data}/O.fvecs"
        elif [ "$randomize" -eq 3 ]
        then 
            echo "HNSW++"
            index="${path}/${data}/P${data}_ef${ef}_M${M}.index"
            trans="${path}/${data}/P.fvecs"
        elif [ "$randomize" -eq 4 ]
        then 
            echo "HNSW+"
            index="${path}/${data}/P${data}_ef${ef}_M${M}.index"
            trans="${path}/${data}/P.fvecs"
        else
            echo "HNSW"
            index="${path}/${data}/${data}_ef${ef}_M${M}.index"    
            trans="${path}/${data}/O.fvecs"
        fi

        res="${result_path}/${data}_HNSW_ef${ef}_M${M}_K${K}_S${sign}_P${gap}_${randomize}.log"
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        lmds="${path}/${data}/LMD.fvecs"
        epsilons="${path}/${data}/E/${sign}.fvecs"

        ./src/search_hnsw -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${lmds} -s ${epsilons} -p ${gap}

    done
done


