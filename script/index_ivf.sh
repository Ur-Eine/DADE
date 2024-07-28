
cd ..
g++ -o ./src/index_ivf ./src/index_ivf.cpp -I ./src/ -O3

C=4096

for data in 'deep1M' 'gist' 'glove2.2m' 'msong' 'tiny5m' 'word2vec'
do 
    for adaptive in $(seq 0 4)
    do

        echo "Indexing - ${data}"

        data_path=./data/${data}
        index_path=./data/${data}

        if [ "$adaptive" -eq 0 ] # raw vectors 
        then
            data_file="${data_path}/${data}_base.fvecs"
            centroid_file="${data_path}/${data}_centroid_${C}.fvecs"
        elif [ "$adaptive" -eq 1 ] || [ "$adaptive" -eq 2 ] # random preprocessed vectors 
        then                 
            data_file="${data_path}/O${data}_base.fvecs"
            centroid_file="${data_path}/O${data}_centroid_${C}.fvecs"
        elif [ "$adaptive" -eq 3 ] || [ "$adaptive" -eq 4 ] # pca preprocessed vectors 
        then                 
            data_file="${data_path}/P${data}_base.fvecs"
            centroid_file="${data_path}/P${data}_centroid_${C}.fvecs"
        fi

        # 0 - IVF, 1,3 - IVF++, 2,4 - IVF+
        index_file="${index_path}/${data}_ivf_${C}_${adaptive}.index"


        ./src/index_ivf -d $data_file -c $centroid_file -i $index_file -a $adaptive
    
    done
done