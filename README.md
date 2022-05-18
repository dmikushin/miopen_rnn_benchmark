# MIOpen RNN Benchmark

MIOpen RNN Benchmark based on [mydeepbench](https://github.com/dmikushin/mydeepbench).

## Current results

AMD Vega56:

```
$ bin/rnn_bench 2>/dev/null
                         Times
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
hidden_size     batch_size     time_steps     rnn_type     fwd_time (usec)  bwd_inputs_time (usec)  bwd_params_time (usec)  total_time (usec)  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 1760             16             50            vanilla       3285                    3197                       0               6482
 1760             32             50            vanilla       3881                    3798                       0               7679
 1760             64             50            vanilla       4895                    4978                       0               9873
 1760            128             50            vanilla       8140                    7925                       0              16065
 2048             16             50            vanilla       5359                    4020                       0               9379
 2048             32             50            vanilla       6422                    5213                       0              11635
 2048             64             50            vanilla       9500                   10293                       0              19793
 2048            128             50            vanilla      16863                   16456                       0              33319
 2560             16             50            vanilla       7579                    5297                       0              12876
 2560             32             50            vanilla       8475                    6847                       0              15322
 2560             64             50            vanilla      13112                   10058                       0              23170
 2560            128             50            vanilla      21337                   15967                       0              37304
  512             16             25               lstm        813                    2255                       0               3068
  512             32             25               lstm       1253                    2390                       0               3643
  512             64             25               lstm       1442                    2593                       0               4035
  512            128             25               lstm       2383                    3159                       0               5542
 1024             16             25               lstm       2464                    2759                       0               5223
 1024             32             25               lstm       2679                    3060                       0               5739
 1024             64             25               lstm       3766                    3709                       0               7475
 1024            128             25               lstm       6399                    8142                       0              14541
 2048             16             25               lstm      13780                    8542                       0              22322
 2048             32             25               lstm      14238                    9307                       0              23545
 2048             64             25               lstm      14871                   13140                       0              28011
 2048            128             25               lstm      23104                   22791                       0              45895
 4096             16             25               lstm      37429                   25962                       0              63391
 4096             32             25               lstm      45531                   32125                       0              77656
 4096             64             25               lstm      60188                   51806                       0             111994
 4096            128             25               lstm      85069                   76273                       0             161342
 1536              8             50               lstm      10745                   10686                       0              21431
 1536             16             50               lstm      12084                   11031                       0              23115
 1536             32             50               lstm      22598                   12040                       0              34638
  256             16            150               lstm       3445                    2455                       0               5900
  256             32            150               lstm       4902                    3200                       0               8102
  256             64            150               lstm       4868                    4714                       0               9582
```

## Current Issues

```
In file included from <built-in>:2:
/opt/rocm-5.1.1/llvm/lib/clang/14.0.0/include/opencl-c.h:5344:26: error: OpenCL extension 'cl_khr_fp64' is core feature or supported optional core feature - ignoring [-Werror,-Wpedantic-core-features]
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
```

```
'+code-object-v3' is not a recognized feature for this target (ignoring feature)
```

```
CHECK_MIOPEN_ERROR(miopenRNNForwardTraining(...)):

MIOpen Error: ../src/ocl/tensorocl.cpp:1404: 
terminate called after throwing an instance of 'std::runtime_error'
  what():  MIOPEN failure: 3 in ./rnn_bench_rocm.cpp at line: 140
```

