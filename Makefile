HIPCC=/opt/rocm/bin/hipcc

all: rnn_bench_rocm

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

rnn_bench_rocm: src/rnn_bench_rocm.cpp
	$(HIPCC) $< -o $@ -I./include -L/opt/rocm/miopen/lib -lMIOpen $(OPT) -std=c++11 --amdgpu-target=gfx900

clean:
	rm -rf rnn_bench

