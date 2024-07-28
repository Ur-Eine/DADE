
cd ..

g++ -o ./src/index_flat ./src/index_flat.cpp -I ./src/ -O3

for data in 'deep1M' 'gist'
do
echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}

data_file="${data_path}/${data}_base.fvecs"
index_file="${index_path}/${data}_flat.index"
./src/index_flat -d $data_file -i $index_file

data_file="${data_path}/O${data}_base.fvecs"
index_file="${index_path}/O${data}_flat.index"
./src/index_flat -d $data_file -i $index_file

data_file="${data_path}/P${data}_base.fvecs"
index_file="${index_path}/P${data}_flat.index"
./src/index_flat -d $data_file -i $index_file
done
