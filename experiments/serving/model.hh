#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include <nccl.h>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

// include swifttransformer headers
#include "model/gpt/gpt_hyper_param.h"
#include "model/gpt/gpt_pagedattn_param.h"
#include "model/gpt/gpt_parallelism_param.h"
#include "model/gpt/gpt.h"

#include "common_gpt_hyper_params.h"

#define ROUND_UP(x, y)  (((x) + (y) - 1) / (y) * (y))

#define NCCL_CHECK(cmd) do { \
  ncclResult_t error = cmd; \
  if (error != ncclSuccess) { \
    fprintf(stderr, "NCCL error in %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(error)); \
    exit(1); \
  } \
} while(0)

using TokenList = std::vector<int64_t>;

struct TestParams {
  TestParams() {}
  std::string model_name;
  std::string model_path;
  std::string vocab_json_path;
  std::string precision;

  int64_t input_len;  // length of input sequence
  int64_t batch_size; // num of input sequences
  int64_t block_size; // how many kv cache entries in a block
  int64_t num_decoding_step;
  int64_t tp_sz;

  int tp_rk;
  ncclUniqueId tp_id, pp_id;
  ncclComm_t inter_stage_comm;
  int rank;
  int world_size;
};

class CudaTimer {
public:
    // 初始化事件
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // 开始计时
    void startTiming() {
        cudaEventRecord(start, 0);
    }

    // 结束计时
    void stopTiming(const std::string& section_name) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        sections.push_back({section_name, milliseconds});
    }

    // 打印所有部分的执行时间
    void printTimes() {
        for (const auto& section : sections) {
            std::cout << "Section " << section.first << " execution time: " << section.second << " ms" << std::endl;
        }
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    cudaEvent_t start, stop;
    std::vector<std::pair<std::string, float>> sections;
};

class TestModel {
public:
  void LoadTestParams(TestParams params) {
    hyper_param = str2hyperparam(params.model_name);

    parallel_param.tensor_para_size = params.tp_sz;
    parallel_param.tensor_para_rank = params.tp_rk;
    parallel_param.init_by_hyper_param(hyper_param);

    pagedattn_param = {
      .block_size = params.block_size,
      .max_num_block_per_req = ROUND_UP(params.input_len + params.num_decoding_step, params.block_size) / params.block_size,
    };

    cuda_device_id = params.tp_rk;
    cuda_device_num = params.tp_sz;
    
    tp_id = params.tp_id;
    pp_id = params.pp_id;

    //num_blocks = pagedattn_param.max_num_block_per_req * params.batch_size;

    context_block_size = ROUND_UP(params.input_len + 1, params.block_size) * params.batch_size / params.block_size;
    max_block_size = ROUND_UP(params.input_len + params.num_decoding_step, params.block_size) * params.batch_size / params.block_size;

    batch_size = params.batch_size;
    inter_stage_comm = params.inter_stage_comm;

    rank = params.rank;
    world_size = params.world_size;

    max_decoding_step = params.num_decoding_step;
  }
  virtual void PreProcessInit() = 0;
  virtual void PreProcess(std::vector<TokenList> &input_tokens) = 0;
  virtual void Forward() = 0;
  virtual double PostProcess() = 0;

  // common members
  st::model::GptHyperParam hyper_param;
  st::model::GptParallelismParam parallel_param;
  st::model::GptPagedAttnParam pagedattn_param;

  int max_decoding_step;

  int cuda_device_id;
  int rank, world_size;
  int cuda_device_num;
  ncclUniqueId tp_id, pp_id;
  int64_t num_blocks; // num of blocks, each block has multiple kv cache entries, specified in pagedattention_param
  int64_t context_block_size; // how many kv cache entries in a block
  int64_t max_block_size;     // how many kv cache entries in a block
  int64_t batch_size;
  ncclComm_t inter_stage_comm;
  cudaStream_t stream;
  float comm_time;
  CudaTimer timer;
};
template <typename T>
void print_kvcache_size_params(size_t context_block_size, size_t local_num_kv_heads, size_t pagedattn_block_size, size_t hyper_num_layers, size_t hyper_head_dim) {
    // size_t cache_sz = sizeof(T) * context_block_size * hyper_num_layers * local_num_kv_heads * pagedattn_block_size * hyper_num_layers * hyper_head_dim;
    
    std::cout << "sizeof(T): " << sizeof(T) << std::endl;
    std::cout << "context_block_size: " << context_block_size << std::endl;
    std::cout << "hyper_param.num_layers: " << hyper_num_layers << std::endl;
    std::cout << "local_num_kv_heads: " << local_num_kv_heads << std::endl;
    std::cout << "pagedattn_param.block_size: " << pagedattn_block_size << std::endl;
    std::cout << "hyper_param.head_dim: " << hyper_head_dim << std::endl;
    // std::cout << "cache_sz: " << cache_sz << std::endl;
}
template<typename T>
class TestContextStageModel : public TestModel {
public:

  void PreProcessInit() override {
    model = new st::model::Gpt<T>(
      hyper_param,
      pagedattn_param,
      parallel_param
    );
    if (parallel_param.tensor_para_size > 1) {
      model->init_communicator(tp_id, pp_id);
    }

    model->initDummyWeight();
    cudaStreamCreate(&stream);
  }
  void PreProcess(std::vector<TokenList> &input_tokens) override {
    // model = new st::model::Gpt<T>(
    //   hyper_param,
    //   pagedattn_param,
    //   parallel_param
    // );
    // if (parallel_param.tensor_para_size > 1) {
    //   model->init_communicator(tp_id, pp_id);
    // }

    // model->initDummyWeight();
    // cudaStreamCreate(&stream);

    std::copy(input_tokens.begin(), input_tokens.end(), std::back_inserter(input_tokens_batched));
    
  }

  void Forward() override {
    timer.startTiming();
    // do something
    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    const int64_t kvcache_sz = context_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    CUDA_CHECK(cudaMalloc(&d_k_cache, (long long)sizeof(T) * kvcache_sz));
    CUDA_CHECK(cudaMalloc(&d_v_cache, (long long)sizeof(T) * kvcache_sz));

    CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req));

    int64_t num_allocated_blocks = 0;
    std::function<int64_t(void)> allocateNewBlock = [&]() -> int64_t {
      num_allocated_blocks += 1;
      if(num_allocated_blocks > max_block_size) {
        printf("Allocating new block %ld, but num_total_blocks is %ld\n", num_allocated_blocks, max_block_size);
        exit(1);
      }
      return num_allocated_blocks;
    };

    allocated_block_cnt = new int64_t[batch_size];
    int64_t* h_block_table = new int64_t[batch_size * pagedattn_param.max_num_block_per_req];

    for (int64_t i = 0; i < batch_size; i++) {
      int64_t block_needed = ROUND_UP(input_tokens_batched[i].size(), pagedattn_param.block_size) / pagedattn_param.block_size;
      allocated_block_cnt[i] = block_needed;
      assert (block_needed <= pagedattn_param.max_num_block_per_req);
      for (int64_t j = 0; j < block_needed; j++) {
        h_block_table[i * pagedattn_param.max_num_block_per_req + j] = allocateNewBlock();
      }
      for (int64_t j = block_needed; j < pagedattn_param.max_num_block_per_req; ++j) {
        h_block_table[i * pagedattn_param.max_num_block_per_req + j] = -10000000;
      }
    }
    CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice));

    delete[] h_block_table;

    cudaDeviceSynchronize();

    // if (parallel_param.tensor_para_size > 1 || parallel_param.pipeline_para_size > 1)
      // MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int64_t> first_token_indexes(batch_size, 0);
    timer.stopTiming("context preparation");

    timer.startTiming();
    context_output_batched = model->forward(
      input_tokens_batched,
      first_token_indexes,
      d_k_cache,
      d_v_cache,
      d_block_table
    );
    timer.stopTiming("context forward computation");
    timer.printTimes();
  }

  double PostProcess() override {
    // send context_output_batched
    int peer = rank + world_size / 2;
    assert(peer < world_size);
    assert(context_output_batched.size() == batch_size);
    printf("context output size: %ld batch size: %ld\n", context_output_batched.size(), batch_size);
    
    int64_t* d_context_output;
    int64_t* d_allocated_block_cnt;
    CUDA_CHECK(cudaMalloc(&d_context_output, sizeof(int64_t) * batch_size));
    CUDA_CHECK(cudaMalloc(&d_allocated_block_cnt, sizeof(int64_t) * batch_size));
    
    CUDA_CHECK(cudaMemcpy(d_context_output, context_output_batched.data(), sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_allocated_block_cnt, allocated_block_cnt, sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    double total_comm_volume = 0; // total communication volume
    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    size_t cache_sz = sizeof(T) * context_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    printf("cache_sz: %zu\n", cache_sz);
    print_kvcache_size_params<T>(context_block_size, local_num_kv_heads, pagedattn_param.block_size, hyper_param.num_layers, hyper_param.head_dim);

    ncclGroupStart();
    NCCL_CHECK(ncclSend((void *)d_context_output, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    // total_comm_volume+=batch_size * sizeof(int64_t);
    // printf("Sent context_output size: %ld bytes\n", batch_size * sizeof(int64_t));
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // send d_k_cache, d_v_cache
    
    NCCL_CHECK(ncclSend(d_k_cache, cache_sz, ncclChar, peer, inter_stage_comm, stream));
    // total_comm_volume+=cache_sz;
    // printf("Sent d_k_cache size: %ld bytes\n", cache_sz);
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    NCCL_CHECK(ncclSend(d_v_cache, cache_sz, ncclChar, peer, inter_stage_comm, stream));
    // printf("Sent d_v_cache size: %ld bytes\n", cache_sz);
    // total_comm_volume+=cache_sz;
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    // send block status, including num_allocated and allocation for each batch
    NCCL_CHECK(ncclSend(d_block_table, batch_size * pagedattn_param.max_num_block_per_req, ncclInt64, peer, inter_stage_comm, stream));
    // size_t block_table_size = sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req;
    // total_comm_volume+=block_table_size;
    // printf("Sent d_block_table size: %ld bytes\n", block_table_size);

    //CUDA_CHECK(cudaStreamSynchronize(stream));
    NCCL_CHECK(ncclSend(d_allocated_block_cnt, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    // total_comm_volume += batch_size * sizeof(int64_t);
    // printf("Sent d_block_table size: %ld bytes\n", batch_size * sizeof(int64_t));
    
    ncclGroupEnd();
    // may need to synchronize
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&comm_time, start, stop);
    total_comm_volume += 1.0 * batch_size * sizeof(int64_t);
    total_comm_volume += 1.0 * cache_sz*2 ;
    total_comm_volume += 1.0 * sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req;
    total_comm_volume += 1.0 * batch_size * sizeof(int64_t);
    printf("send time: %f ms, comm volume: %lf \n ", comm_time, total_comm_volume);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free all resources
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_context_output));
    CUDA_CHECK(cudaFree(d_allocated_block_cnt));

    delete model;
    CUDA_CHECK(cudaStreamDestroy(stream));

    return comm_time;
  }

// models with templates
  st::model::Gpt<T> *model;
  T* *d_k_cache, *d_v_cache;
  std::vector<TokenList> input_tokens_batched;
  std::vector<int64_t> context_output_batched;
  int64_t* d_block_table;
  int64_t* allocated_block_cnt;
};


template<typename T>
class TestDecodingStageModel : public TestModel {
public:
  T *d_k_cache, *d_v_cache;
  st::model::Gpt<T> *model;
  int64_t* d_block_table;
  int64_t num_allocated_blocks;
  std::vector<std::vector<int64_t>> output_tokens_batched;
  std::vector<std::vector<int64_t>> next_iter_input_tokens_batched; 

  std::vector<int64_t> first_token_indexes;
  std::vector<int64_t> input_tokens_size;
  int64_t* allocated_block_cnt;

  int64_t allocateNewBlock() {
    num_allocated_blocks += 1;
    if (num_allocated_blocks >= max_block_size) {
        printf("Allocating new block %ld, but num_total_blocks is %ld\n", num_allocated_blocks, max_block_size);
        exit(1);
    }
    return num_allocated_blocks;
  }

  void PreProcessInit() override  {
    // init the model only init it once
    model = new st::model::Gpt<T>(
      hyper_param,
      pagedattn_param,
      parallel_param
    );
    model->initDummyWeight();
    cudaStreamCreate(&stream);

    if (parallel_param.tensor_para_size > 1) {
      model->init_communicator(tp_id, pp_id);
    }
  }

  void PreProcess(std::vector<TokenList> &input_tokens) override {
    // syj here still need to consider how to change the info based on the new arrive request
    int peer = rank - world_size / 2;
    output_tokens_batched.resize(batch_size);
    next_iter_input_tokens_batched.resize(batch_size);
    first_token_indexes.resize(batch_size);
    input_tokens_size.resize(batch_size);

    int64_t *context_output_tokens;
    CUDA_CHECK(cudaMallocManaged(&context_output_tokens, sizeof(int64_t) * batch_size));
    
    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    int64_t num_total_blocks = pagedattn_param.max_num_block_per_req * batch_size + 1;
    size_t cache_sz = sizeof(T) * max_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    CUDA_CHECK(cudaMalloc(&d_k_cache, cache_sz));
    CUDA_CHECK(cudaMalloc(&d_v_cache, cache_sz));
    CUDA_CHECK(cudaDeviceSynchronize());

    num_allocated_blocks = 0;
    allocated_block_cnt = new int64_t[batch_size];
    int64_t* d_allocated_block_cnt;
    CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req));
    CUDA_CHECK(cudaMalloc(&d_allocated_block_cnt, sizeof(int64_t) * batch_size));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    size_t context_cache_sz = sizeof(T) * context_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    // recv from context stage
    ncclGroupStart();
    NCCL_CHECK(ncclRecv(context_output_tokens, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_k_cache, context_cache_sz, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_v_cache, context_cache_sz, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void*)d_block_table, batch_size * pagedattn_param.max_num_block_per_req, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void*)d_allocated_block_cnt, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    ncclGroupEnd();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaMemcpyAsync(allocated_block_cnt, d_allocated_block_cnt, sizeof(int64_t) * batch_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    cudaEventElapsedTime(&comm_time, start, stop);
    printf("recv time: %f ms\n", comm_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // init output_tokens_batched;
    for (uint i = 0; i < batch_size; i++)
      output_tokens_batched[i].push_back(context_output_tokens[i]);
    for (uint i = 0; i < batch_size; i++)
      next_iter_input_tokens_batched[i].push_back(context_output_tokens[i]);
    // when first_token_indexes[i] > 0, the gpt will take this as the first output token
    for (uint i = 0; i < batch_size; i++)
      first_token_indexes[i] = input_tokens[i].size();
    // initialize input_tokens_size
    for (int64_t i = 0; i < batch_size; i++) {
      input_tokens_size[i] = first_token_indexes[i];
    }

    CUDA_CHECK(cudaFree(context_output_tokens));
    CUDA_CHECK(cudaFree(d_allocated_block_cnt));
  }

  void Forward() override {
    timer.startTiming();
    int64_t cur_notdone_input_cnt = batch_size;
    int64_t* request_id = new int64_t[batch_size];
    const int64_t end_token = -1;
    
    for (int64_t i = 0; i < batch_size; i++) {
      request_id[i] = i;
    }

    for (int64_t step = 0; step < max_decoding_step && cur_notdone_input_cnt > 0 ; step++) {
      // Allocate new blocks if needed
      for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
        // todo: check if output_tokens_batched is initialized
        int64_t block_needed = ROUND_UP(input_tokens_size[request_id[i]]
                    + output_tokens_batched[request_id[i]].size(), pagedattn_param.block_size) / pagedattn_param.block_size;
        while (allocated_block_cnt[request_id[i]] < block_needed) {
          int64_t new_block = allocateNewBlock();
          CUDA_CHECK(cudaMemcpy(d_block_table + i*pagedattn_param.max_num_block_per_req + allocated_block_cnt[request_id[i]],
            &new_block, sizeof(int64_t), cudaMemcpyHostToDevice));
          allocated_block_cnt[request_id[i]] += 1;
        }
      }
      // sync_check_cuda_error();
      cudaDeviceSynchronize();

      auto cur_iter_output_tokens = model->forward(
        next_iter_input_tokens_batched,
        first_token_indexes,

        d_k_cache,
        d_v_cache,
        d_block_table	
      );
      // Decode generated token and put them into output_tokens_batched
      // Prepare input for the next round
      int64_t ptr = 0;
      int64_t new_notdone_input_cnt = cur_notdone_input_cnt;
      next_iter_input_tokens_batched.clear();
      first_token_indexes.clear();
      for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
        int64_t result_token = cur_iter_output_tokens[i];
        output_tokens_batched[request_id[i]].push_back(result_token);
        if (result_token == end_token) {
          // The generation of this request is done
          --new_notdone_input_cnt;
        } else {
          next_iter_input_tokens_batched.push_back(std::vector<int64_t>{result_token});
          first_token_indexes.push_back(input_tokens_size[request_id[i]] + output_tokens_batched[request_id[i]].size() - 1);
          // Copy k/v cache to the right place if necessary (can be optimized in the future)
          if (i != ptr) {
            request_id[ptr] = request_id[i];
            cudaMemcpyAsync(
              d_block_table + ptr*pagedattn_param.max_num_block_per_req,
              d_block_table + i*pagedattn_param.max_num_block_per_req,
              sizeof(int64_t) * pagedattn_param.max_num_block_per_req,
              cudaMemcpyDeviceToDevice
            );
            // sync_check_cuda_error();
            cudaDeviceSynchronize();
          }
          ptr += 1;
        }
      }
      cur_notdone_input_cnt = new_notdone_input_cnt;
    }
    timer.stopTiming("decode forward computation");
    timer.printTimes();
    delete[] request_id;
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_block_table);
  }

  double PostProcess() override {
    // free resources
    delete[] allocated_block_cnt;
    delete model;
    cudaStreamDestroy(stream);
    return comm_time;
  }
};

#endif