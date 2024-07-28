

cd ..

g++ ./src/search_flat.cpp -O3 -o ./src/search_flat -I ./src/

path=./data/
result_path=./results/

K=100

for gap in 1 32
do
    for data in 'deep1M' 'gist'
    do
        for randomize in $(seq 1 2)
        do
            if [ "$randomize" -eq 1 ]
            then 
                echo "FLAT++"
                index="${path}/${data}/O${data}_flat.index"
                trans="${path}/${data}/O.fvecs"
            elif [ "$randomize" -eq 2 ]
            then 
                echo "FLAT**"
                index="${path}/${data}/P${data}_flat.index"
                trans="${path}/${data}/P.fvecs"
            else
                echo "FLAT"
                index="${path}/${data}/${data}_flat.index"    
                trans="${path}/${data}/O.fvecs"
            fi

            res="${result_path}/${data}_FLAT_K${K}_P${gap}_${randomize}.log"
            query="${path}/${data}/${data}_query.fvecs"
            gnd="${path}/${data}/${data}_groundtruth.ivecs"
            lmds="${path}/${data}/LMD.fvecs"
            epsilons="${path}/${data}/E/"

            ./src/search_flat -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${lmds} -s ${epsilons} -p ${gap}

        done
    done
done

for data in 'deep1M' 'gist'
do
    for randomize in $(seq 3 4)
    do
        if [ "$randomize" -eq 3 ]
        then 
            echo "FLAT RANDOM"
            index="${path}/${data}/O${data}_flat.index"
            trans="${path}/${data}/O.fvecs"
        elif [ "$randomize" -eq 4 ]
        then
            echo "FLAT PCA"
            index="${path}/${data}/P${data}_flat.index"
            trans="${path}/${data}/P.fvecs"
        fi

        res="${result_path}/${data}_FLAT_K${K}_P0_${randomize}.log"
        query="${path}/${data}/${data}_query.fvecs"
        gnd="${path}/${data}/${data}_groundtruth.ivecs"
        lmds="${path}/${data}/LMD.fvecs"
        epsilons="${path}/${data}/E/"

        ./src/search_flat -d ${randomize} -n ${data} -i ${index} -q ${query} -g ${gnd} -r ${res} -t ${trans} -k ${K} -l ${lmds} -s ${epsilons}

    done
done

