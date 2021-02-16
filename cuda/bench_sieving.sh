DEFINES="-Xcompiler -DMAX_SIEVING_DIM=160 -Xcompiler -DGPUVECNUM=131072 -Xcompiler -DHAVE_CUDA -I../parallel-hashmap" #-Xcompiler -DDEBUG_BENCHMARK"

if [ -z "$1" ]
then
    /usr/local/cuda/bin/nvcc -ccbin g++ -Xcompiler -fPIC -Xcompiler -Ofast -Xcompiler -march=native -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wextra $DEFINES -std=c++11 -O3  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -lineinfo -I/usr/local/cuda/include -c ../cuda/GPUStreamGeneral.cu -o GPUStreamGeneral.o
fi

/usr/local/cuda/bin/nvcc -ccbin g++ -Xcompiler -fPIC -Xcompiler -Ofast -Xcompiler -march=native -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wextra $DEFINES -std=c++11 -O3  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -lineinfo -I/usr/local/cuda/include -lcublas -lcurand --resource-usage bench_sieving.cpp -o bench_sieving GPUStreamGeneral.o

/usr/local/cuda/bin/nvcc -ccbin g++ -Xcompiler -fPIC -Xcompiler -Ofast -Xcompiler -march=native -Xcompiler -pthread -Xcompiler -Wall -Xcompiler -Wextra $DEFINES -std=c++11 -O3  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -lineinfo -I/usr/local/cuda/include -lcublas -lcurand --resource-usage bench_quality.cpp -o bench_quality GPUStreamGeneral.o
