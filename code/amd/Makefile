SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p

#hipcc
HIPCC=/opt/rocm/bin/hipcc

#BLAS
ROCBLAS_LIB=rocblas
ROCBLAS_LIBPATH=/opt/rocm/rocblas/lib

#CONV
MIOPEN_LIB?=MIOpen
MIOPEN_LIBPATH?=/opt/rocm/miopen/lib

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels

all: conv rnn gemm 

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(DEEPBENCH_INC) -L$(MIOPEN_LIBPATH) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

rnn:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $(BIN_DIR)/rnn_bench -I$(DEEPBENCH_INC) -L$(MIOPEN_LIBPATH)  -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(DEEPBENCH_INC) -L$(ROCBLAS_LIBPATH) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

clean:
	rm -rf $(BIN_DIR)

