
# 0 - IVF, 1 - IVF++, 2 - IVF+
randomize=0

cd ..
g++ ./src/search_ivf.cpp -O3 -o ./src/search_ivf -I ./src/

path=./data/
result_path=./results

C=4096
K=20
sign=0.10
gap=32

for data in 'deep1M' 'gist' 'glove2.2m' 'msong' 'tiny5m' 'word2vec'
do 
    for randomize in $(seq 0 4)
    do

        if [ "$randomize" -eq 1 ]
        then 
            echo "IVF++"
            trans="${path}/${data}/O.fvecs"
        elif [ "$randomize" -eq 2 ]
        then 
            echo "IVF+"
            trans="${path}/${data}/O.fvecs"
        elif [ "$randomize" -eq 3 ]
        then 
            echo "IVF++"
            trans="${path}/${data}/P.fvecs"
        elif [ "$randomize" -eq 4 ]
        then 
            echo "IVF+"
            trans="${path}/${data}/P.fvecs"
        else
            echo "IVF"
            trans="${path}/${data}/O.fvecs"
        fi

        res="${result_path}/${data}_IVF${C}_K${K}_S${sign}_P${gap}_${randomize}.log"
        index="${path}/${data}/${data}_ivf_${C}_${randomize}.index"

        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        lmds="${path}/${data}/LMD.fvecs"
        epsilons="${path}/${data}/E/${sign}.fvecs"

        ./src/search_ivf -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${lmds} -s ${epsilons} -p ${gap} 

    done
done
