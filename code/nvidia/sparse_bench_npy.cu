#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <cnpy.h>
#include "tensor.h"
#include "cuda_helper.h"
#include "sparse_gemm_problems.h"

curandGenerator_t curand_gen;

std::string getCusparseErrorString(cusparseStatus_t status) {
    if (status == CUSPARSE_STATUS_SUCCESS)
        return "CUSPARSE_STATUS_SUCCESS";
    else if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    else if (status == CUSPARSE_STATUS_MAPPING_ERROR)
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    else if (status == CUSPARSE_STATUS_EXECUTION_FAILED)
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    else if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    else if (status == CUSPARSE_STATUS_ZERO_PIVOT)
        return "CUSPARSE_STATUS_ZERO_PIVOT";
    else
        return "Unknown CUSPARSE error type";
}

void throw_cusparse_err(cusparseStatus_t status, int line, const char* filename) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "CUSPARSE failure: " << getCusparseErrorString(status) <<
              " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUSPARSE_ERROR(status) throw_cusparse_err(status, __LINE__, __FILE__)

class CusparseHandle {
    std::shared_ptr<cusparseHandle_t> ptr_;

    struct CusparseHandleDeleter {
        void operator()(cusparseHandle_t * handle) {
            cusparseDestroy(*handle);
        }
    };

public:

    CusparseHandle() : ptr_(new cusparseHandle_t, CusparseHandleDeleter()) {
        if (cusparseCreate(ptr_.get()) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cusparseHandle");
        }

    }

    cusparseHandle_t  get() { return *ptr_; };
};

class CusparseMatDesc {
    std::shared_ptr<cusparseMatDescr_t> ptr_;

    struct CusparseMatDescDeleter {
        void operator()(cusparseMatDescr_t * handle) {
            cusparseDestroyMatDescr(*handle);
        }
    };

    public:

    CusparseMatDesc() : ptr_(new cusparseMatDescr_t, CusparseMatDescDeleter()) {
        if (cusparseCreateMatDescr(ptr_.get()) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cusparse matrix descriptor");
        }

    }

    cusparseMatDescr_t get() { return *ptr_; };

};

template <typename T>
class SparseOp {

    Tensor<T> csrVal_;
    Tensor<int> csrRowPtr_;
    Tensor<int> csrColInd_;

    Tensor<int> nnzPerRow_;

    Tensor<int> cooRow_;

    int rows_, cols_;
    int nnzTotal_;

    // Cusparse handle
    CusparseHandle cusparse_handle_;

    // Cusparse matrix descriptor.
    CusparseMatDesc cusparse_descr_;


    float alpha_ = 1.0;
    float beta_ = 0.0;

public:

    SparseOp(int rows, int cols, Tensor<T> dense_array) :
        rows_(rows),
        cols_(cols),
        cusparse_handle_(),
        cusparse_descr_() {


        //Assign descr attributes
        cusparseSetMatType(cusparse_descr_.get(),CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(cusparse_descr_.get(),CUSPARSE_INDEX_BASE_ZERO);

        nnzPerRow_ = Tensor<int>({rows_});

        // compute total non zeros and non zeros per row.
        CHECK_CUSPARSE_ERROR(cusparseSnnz(cusparse_handle_.get(),
                                          CUSPARSE_DIRECTION_ROW,
                                          rows_,
                                          cols_,
                                          cusparse_descr_.get(),
                                          dense_array.begin(),
                                          rows_,
                                          nnzPerRow_.begin(),
                                          &nnzTotal_));

        //allocate memory for csr matrix arrays.
        csrVal_ = Tensor<T>({nnzTotal_});
        csrRowPtr_ = Tensor<int>({rows_+1});
        csrColInd_ = Tensor<int>({nnzTotal_});
        cooRow_ = Tensor<int>({nnzTotal_});

        // Convert Dense matrix to CSR format
        CHECK_CUSPARSE_ERROR(cusparseSdense2csr(cusparse_handle_.get(),
                                                rows_,
                                                cols_,
                                                cusparse_descr_.get(),
                                                dense_array.begin(),
                                                rows_,
                                                nnzPerRow_.begin(),
                                                csrVal_.begin(),
                                                csrRowPtr_.begin(),
                                                csrColInd_.begin()));

        CHECK_CUSPARSE_ERROR(cusparseXcsr2coo(cusparse_handle_.get(),
                                                csrRowPtr_.begin(),
                                                nnzTotal_,
                                                rows_,
                                                cooRow_.begin(),
                                                CUSPARSE_INDEX_BASE_ZERO));

    }

    float sparse_gemm(Tensor<float> b, Tensor<float> c, int mini_batch) {

        if (mini_batch == 1) {
            // Sparse gemv
            CHECK_CUSPARSE_ERROR(cusparseScsrmv(cusparse_handle_.get(),
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                rows_,
                                                cols_,
                                                nnzTotal_,
                                                &alpha_,
                                                cusparse_descr_.get(),
                                                csrVal_.begin(),
                                                csrRowPtr_.begin(),
                                                csrColInd_.begin(),
                                                b.begin(),
                                                &beta_,
                                                c.begin()));
             return 0.0;
        } else {
            // Sparse gemm
            void * dBuffer = NULL;
            size_t bufferSize = 0;
//            CHECK_CUSPARSE_ERROR(cusparseScsrmm(cusparse_handle_.get(),
//                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                                rows_,
//                                                mini_batch,
//                                                cols_,
//                                                nnzTotal_,
//                                                &alpha_,
//                                                cusparse_descr_.get(),
//                                                csrVal_.begin(),
//                                                csrRowPtr_.begin(),
//                                                csrColInd_.begin(),
//                                                b.begin(),
//                                                cols_,
//                                                &beta_,
//                                                c.begin(),
//                                                rows_));

            cusparseDnMatDescr_t B,C;
            cusparseSpMatDescr_t A;

            CHECK_CUSPARSE_ERROR(cusparseCreateCoo(&A,
            rows_,
            cols_,
            nnzTotal_,
            cooRow_.begin(),
            csrColInd_.begin(),
            csrVal_.begin(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
            )
            );

//            CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&A,
//            rows_,
//            cols_,
//            nnzTotal_,
//            csrRowPtr_.begin(),
//            csrColInd_.begin(),
//            csrVal_.begin(),
//            CUSPARSE_INDEX_32I,
//            CUSPARSE_INDEX_32I,
//            CUSPARSE_INDEX_BASE_ZERO,
//            CUDA_R_32F
//            )
//            );

            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B,
            b.dims()[1],
            b.dims()[0],
            b.dims()[1], //ld
            b.begin(),
            CUDA_R_32F,
            CUSPARSE_ORDER_COL
            )
            );

            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C,
            c.dims()[0],
            c.dims()[1],
            c.dims()[0], //ld
            c.begin(),
            CUDA_R_32F,
            CUSPARSE_ORDER_COL
            )
            );


            CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(cusparse_handle_.get(),
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             CUSPARSE_OPERATION_TRANSPOSE,
             &alpha_,
             A,
             B,
             &beta_,
             C,
             CUDA_R_32F,
             CUSPARSE_COOMM_ALG3,
             &bufferSize)
             );

             cudaMalloc(&dBuffer, bufferSize);

            for(int i = 0;i < 10; i ++){
                CHECK_CUSPARSE_ERROR(cusparseSpMM(cusparse_handle_.get(),
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 &alpha_,
                 A,
                 B,
                 &beta_,
                 C,
                 CUDA_R_32F,
                 CUSPARSE_COOMM_ALG1,
                 dBuffer)
                                        );}

            cudaDeviceSynchronize();
            auto start = std::chrono::steady_clock::now();

            for(int i = 0;i < 100; i ++){
                CHECK_CUSPARSE_ERROR(cusparseSpMM(cusparse_handle_.get(),
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 &alpha_,
                 A,
                 B,
                 &beta_,
                 C,
                 CUDA_R_32F,
                 CUSPARSE_COOMM_ALG1,
                 dBuffer)
                                        );}

             cudaDeviceSynchronize();
             auto end = std::chrono::steady_clock::now();
            float sp_time = static_cast<float>(std::chrono::duration<double, std::micro>(end - start).count()) / 100.0;

             cudaFree(dBuffer);
             return sp_time;
        }

    }

};

Tensor<float> generate_sparse_matrix_from_npy(std::string filename) {


    cnpy::NpyArray arr = cnpy::npy_load(filename);
    float * cpu_sp_mat = arr.data<float>();
    assert(arr.word_size == sizeof(float));

    int B_dim = arr.shape[0];
    int A_dim = arr.shape[1];

    auto dense_mat = rand<float>({A_dim, B_dim}, curand_gen);


    CHECK_CUDA_ERROR(cudaMemcpy(dense_mat.begin(), cpu_sp_mat, B_dim * A_dim*sizeof(float), cudaMemcpyHostToDevice));

    //delete[] cpu_sp_mat;

    return dense_mat;


}

Tensor<float> generate_sparse_matrix(int rows, int cols, float sparsity) {

    auto dense_mat = rand<float>({rows, cols}, curand_gen);

    float * cpu_sp_mat;

    cpu_sp_mat = new float [rows*cols];

    CHECK_CUDA_ERROR(cudaMemcpy(cpu_sp_mat, dense_mat.begin(), rows*cols*sizeof(float), cudaMemcpyDeviceToHost));

    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (cpu_sp_mat[i*cols+j] < sparsity) {
                cpu_sp_mat[i*cols+j] = 0.f;
                count++;
            }
        }
    }

    CHECK_CUDA_ERROR(cudaMemcpy(dense_mat.begin(), cpu_sp_mat, rows*cols*sizeof(float), cudaMemcpyHostToDevice));

    delete[] cpu_sp_mat;

    return dense_mat;
}

std::tuple<int, int> time_sparse_gemm(Tensor<float> A, Tensor<float> B, Tensor<float> C,
                                      bool a_t, bool b_t, cublasHandle_t cublas_handle) {

    auto a_dims = A.dims();
    auto b_dims = B.dims();

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    const float alpha = 1.f; // / static_cast<float>(A.dims()[1]);
    const float beta  = 1.f;

    int numRepeats = 1;//std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
//    cublasStatus_t stat = cublasSgemm(cublas_handle,
//                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
//                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
//                m,
//                n,
//                k,
//                &alpha,
//                A.begin(), A.dims()[0],
//                B.begin(), B.dims()[0],
//                &beta,
//                C.begin(), C.dims()[0]);
//    if (stat != CUBLAS_STATUS_SUCCESS) {
//        throw std::runtime_error("sgemm failed");
//    }
//
//    cudaDeviceSynchronize();
//
//    auto start = std::chrono::steady_clock::now();
//
//    for (int i = 0; i < numRepeats; ++i) {
//        cublasStatus_t stat = cublasSgemm(cublas_handle,
//                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
//                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
//                    m,
//                    n,
//                    k,
//                    &alpha,
//                    A.begin(), A.dims()[0],
//                    B.begin(), B.dims()[0],
//                    &beta,
//                    C.begin(), C.dims()[0]);
//        if (stat != CUBLAS_STATUS_SUCCESS) {
//            throw std::runtime_error("sgemm failed");
//        }
//    }
//    cudaDeviceSynchronize();
//
//    auto end = std::chrono::steady_clock::now();
//
//    int d_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

    int d_time = 1;
    // Only support 2D arrays in this benchmark.
    assert (a_dims.size() == 2);
    assert (b_dims.size() == 2);

    auto csr_A = SparseOp<float>(a_dims[0], a_dims[1], A);

    float total_time = 0.0;

    csr_A.sparse_gemm(B, C, b_dims[1]);
    cudaDeviceSynchronize();

    //auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        total_time += csr_A.sparse_gemm(B, C, b_dims[1]);
    }

    cudaDeviceSynchronize();
    //auto end = std::chrono::steady_clock::now();
    float sp_time = total_time / numRepeats;


    return std::tuple<float, int>(sp_time, d_time);
}

int main() {

    cudaFree(0);

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failed" << std::endl;
    }


    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t    sparsity   sparse time (usec)  dense time (usec)  speedup" << std::endl;
    for (const auto &problem : inference_server_set) {
        int m,n,k;
        bool a_t, b_t;
        float sparsity;
        std::string filename;

        std::tie(m, k, n, a_t, b_t, sparsity,filename) = problem;

        auto a = generate_sparse_matrix_from_npy(filename);
        auto b = rand<float>({k, n}, curand_gen);
        auto c = zeros<float>({m, n});

        a_t = true;

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << a_t ? "t" : "n";
        std::cout << std::setw(7) << b_t ? "t" : "n";
        std::cout << std::setw(11) << sparsity;
        std::cout << std::setw(13) << std::setprecision(10);

        float sp_time, d_time;
        std::tie(sp_time, d_time) = time_sparse_gemm(a, b, c, a_t, b_t, cublas_handle);

        std::cout << std::setprecision(10) << std::setw(16) << sp_time << std::setprecision(10);

        std::cout << std::setw(16) << d_time;
        std::cout << std::setw(20) << float(d_time)/sp_time;
        std::cout << std::endl;
    }

    return 0;
}
