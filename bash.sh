cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc -B build
cmake --build build --config Release --target demo -j 8 

# Run it
./build/bin/demo


# cmake -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc