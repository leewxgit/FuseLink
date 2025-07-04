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
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <chrono>



#define ROUND_UP(x, y)  (((x) + (y) - 1) / (y) * (y))

#define NCCL_CHECK(cmd) do { \
  ncclResult_t error = cmd; \
  if (error != ncclSuccess) { \
    fprintf(stderr, "NCCL error in %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(error)); \
    exit(1); \
  } \
} while(0)

using TokenList = std::vector<int64_t>;

struct Request {
    double timestamp;
    int prompt_len;
    int active_prompt_len;
    int outputlen;
};

std::queue<Request> request_queue;
std::mutex queueMutex;
std::condition_variable queue_cv;

void readRequests(const std::string& filename) {
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
  }
  std::string line;
  auto start_time = std::chrono::steady_clock::now();

  while (std::getline(inputFile, line)) {
      std::istringstream lineStream(line);
      double timestamp;
      int prompt_length;
      int shared_prompt_length;
      int output_length;
      std::string token;

      if (std::getline(lineStream, token, ':')) {
          lineStream.ignore(1); // Ignore the space after ':'
          lineStream >> timestamp;
      }
      if (std::getline(lineStream, token, ':')) {
          lineStream.ignore(1); // Ignore the space after ':'
          lineStream >> prompt_length;
      }
      if (std::getline(lineStream, token, ':')) {
          lineStream.ignore(1); // Ignore the space after ':'
          lineStream >> shared_prompt_length;
      }
      if (std::getline(lineStream, token, ':')) {
          lineStream.ignore(1); // Ignore the space after ':'
          lineStream >> output_length;
      }
      auto current_time = std::chrono::steady_clock::now();
      // Wait until the timestamp time has elapsed
      double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time).count(); // Sec
      if (timestamp > elapsed) {
          std::this_thread::sleep_for(std::chrono::duration<double>(timestamp - elapsed));
      }

      Request req{timestamp, prompt_length,prompt_length - shared_prompt_length, output_length};
      {
          std::lock_guard<std::mutex> lock(queueMutex);
          request_queue.push(req);
      }
      queue_cv.notify_one();  // Notify the consumer thread
  }
}



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
    cudaEventCreate(&init);
    cudaEventRecord(init, 0);
  }

  // 开始计时
  void startTiming() {
    cudaEventRecord(start, 0);
  }

  // 结束计时
  void stopTiming(const std::string& section_name) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float init2start = .0, init2stop = .0, start2stop = .0;
    cudaEventElapsedTime(&start2stop, start, stop);
    sections.push_back({section_name, {init2start, start2stop, init2stop}});
  }

  // 打印所有部分的执行时间
  void printTimes() {
    for (const auto& section : sections) {
      std::cout << "Section " << section.first << " times: ";
      for (const auto& time : section.second) {
        std::cout << time << " ";
      }
      std::cout << "ms" << std::endl;
    }
  }

  void printLast() {
    std::cout << "Section " << sections.back().first << " times: ";
    for (const auto& time : sections.back().second) {
      std::cout << time << " ";
    }
    std::cout << "ms" << std::endl;
  }

  double getElapsedTime(cudaEvent_t e) {
    float time;
    cudaEventElapsedTime(&time, init, e);
    return time;
  }

  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(init);
  }

private:
  cudaEvent_t init, start, stop;
  std::vector<std::pair<std::string, std::vector<float>>> sections;
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

    max_batch_size = params.batch_size; // max_blocksize

    inter_stage_comm = params.inter_stage_comm;

    rank = params.rank;
    world_size = params.world_size;

    max_decoding_step = params.num_decoding_step;
  }
  virtual void PreProcessInit() = 0;
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
  int64_t max_batch_size; //TODO: all the init batchsize should be this one 
  int64_t total_block;
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
    // printf("stream created, address: %p\n", stream);
  }
  void PreProcess(std::vector<TokenList> &input_tokens, std::vector<int> &input_lens) {
    //get new input tokens and update model?
    if(input_tokens_batched.size()!=0)
      input_tokens_batched.clear(); //clear history input_tokens
    std::copy(input_tokens.begin(), input_tokens.end(), std::back_inserter(input_tokens_batched)); // copy it into the std::vector<TokenList>
    batch_size = input_tokens.size();
    this->input_lens = input_lens;
  }

  void Forward() override {
    timer.startTiming();
    // do something
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

    total_block=0;
    for(int64_t i = 0; i < batch_size; i++) {
      //calculate total block, reserve 5 extra token
      total_block += ROUND_UP(input_lens[i] + 5, pagedattn_param.block_size) / pagedattn_param.block_size; 
    }

    // int64_t* h_block_table = new int64_t[total_block];
    int64_t* h_block_table = new int64_t[max_batch_size * pagedattn_param.max_num_block_per_req];
    int tmp_ctr=0;
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t block_needed = ROUND_UP(input_tokens_batched[i].size(), pagedattn_param.block_size) / pagedattn_param.block_size;
      allocated_block_cnt[i] = block_needed;
      assert (block_needed <= pagedattn_param.max_num_block_per_req);
      for (int64_t j = 0; j < block_needed; j++) {
        h_block_table[i * pagedattn_param.max_num_block_per_req + j] = allocateNewBlock();
      }
      // for (int64_t j = 0; j < block_needed; j++) {
      //   h_block_table[tmp_ctr++] = allocateNewBlock();
      // }
      for (int64_t j = block_needed; j < pagedattn_param.max_num_block_per_req; ++j) {
        h_block_table[i * pagedattn_param.max_num_block_per_req + j] = -10000000;
      }
    }
    // for(int64_t i=0; i< total_block;i++){
    //   if (h_block_table[i]==0 && i!=0) {
    //     h_block_table[i] = -10000000;
    //   }
    // }

    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    const int64_t kvcache_sz = (context_block_size) * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    
    sync_check_cuda_error();
    CUDA_CHECK(cudaMalloc(&d_k_cache, (long long)sizeof(T) * kvcache_sz));
    CUDA_CHECK(cudaMalloc(&d_v_cache, (long long)sizeof(T) * kvcache_sz));
    // CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * context_block_size));
    CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * max_batch_size * pagedattn_param.max_num_block_per_req));
    CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table, sizeof(int64_t) * max_batch_size * pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table, sizeof(int64_t) * (context_block_size), cudaMemcpyHostToDevice));
    delete[] h_block_table;

    cudaDeviceSynchronize();

    std::vector<int64_t> first_token_indexes(batch_size, 0);
    timer.stopTiming("context preparation");

    timer.startTiming();
    sync_check_cuda_error();
    context_output_batched = model->forward(
      input_tokens_batched,
      first_token_indexes,
      d_k_cache,
      d_v_cache,
      d_block_table
    );
    timer.stopTiming("context forward computation");
    timer.printLast();
  }

  double PostProcess() override {
    timer.startTiming();
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
    // printf("sync to stream %p\n", stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Serialize input_tokens_batched to a contiguous buffer
    size_t total_tokens_size = 0;
    for (const auto& token_list : input_tokens_batched) {
        total_tokens_size += token_list.size();
    }
    int64_t* serialized_tokens;
    CUDA_CHECK(cudaMalloc(&serialized_tokens, sizeof(int64_t) * total_tokens_size));
    int64_t* host_serialized_tokens = new int64_t[total_tokens_size];
    size_t offset = 0;
    for (const auto& token_list : input_tokens_batched) {
        memcpy(host_serialized_tokens + offset, token_list.data(), sizeof(int64_t) * token_list.size());
        offset += token_list.size();
    }
    CUDA_CHECK(cudaMemcpy(serialized_tokens, host_serialized_tokens, sizeof(int64_t) * total_tokens_size, cudaMemcpyHostToDevice));
    delete[] host_serialized_tokens;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    uint64_t total_comm_volume = 0; // total communication volume
    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    //TODO: change the volume into total promptlen, now the volume is active prompt len
    size_t cache_sz = sizeof(T) * total_block * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    printf("cache_sz: %zu\n", cache_sz);
    print_kvcache_size_params<T>(total_block, local_num_kv_heads, pagedattn_param.block_size, hyper_param.num_layers, hyper_param.head_dim);

    // Send data size first
    int64_t host_context_output_size = batch_size;
    int64_t host_d_k_cache_size = cache_sz;
    int64_t host_d_v_cache_size = cache_sz;
    int64_t host_block_table_size = max_batch_size * pagedattn_param.max_num_block_per_req;

    
    int64_t *context_output_size;
    int64_t *d_k_cache_size;
    int64_t *d_v_cache_size;
    int64_t *block_table_size;

    CUDA_CHECK(cudaMalloc(&context_output_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_k_cache_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_v_cache_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&block_table_size, sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(context_output_size, &host_context_output_size, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache_size, &host_d_k_cache_size, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache_size, &host_d_v_cache_size, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(block_table_size, &host_block_table_size, sizeof(int64_t), cudaMemcpyHostToDevice));


    //int64_t block_table_size = total_block;
    int64_t* input_tokens_batched_size; //= total_tokens_size;
    CUDA_CHECK(cudaMalloc(&input_tokens_batched_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(input_tokens_batched_size, &total_tokens_size, sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t* host_input_tokens_sizes = new int64_t[20]; //max batch = 20
    for (int i = 0; i < 20; ++i) {
      if(i<batch_size)
        host_input_tokens_sizes[i] = input_tokens_batched[i].size();
      else{
        host_input_tokens_sizes[i]=0;
      }
    }
    int64_t* input_tokens_sizes;
    CUDA_CHECK(cudaMalloc(&input_tokens_sizes, 20 * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(input_tokens_sizes, host_input_tokens_sizes, 20 * sizeof(int64_t), cudaMemcpyHostToDevice));

    timer.stopTiming("context prepare send");

    timer.startTiming();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    ncclGroupStart();
    ncclResult_t res;
    NCCL_CHECK(ncclSend(context_output_size, 1, ncclInt64, peer, inter_stage_comm, stream));//&context_output_size
    NCCL_CHECK(ncclSend(d_k_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(d_v_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(block_table_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(input_tokens_batched_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(input_tokens_sizes, 20, ncclInt64, peer, inter_stage_comm, stream));
    ncclGroupEnd();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    timer.stopTiming("context send size");
    timer.startTiming();
    // Send actual data
    ncclGroupStart();
    NCCL_CHECK(ncclSend(d_context_output, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(d_k_cache, cache_sz, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(d_v_cache, cache_sz, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(d_block_table, host_block_table_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(d_allocated_block_cnt, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclSend(serialized_tokens, total_tokens_size, ncclInt64, peer, inter_stage_comm, stream));
    ncclGroupEnd();
    timer.stopTiming("context finish send");
    // timer.printTimes();
    // Synchronize
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&comm_time, start, stop);
    
    uint64_t comm_volume_1 = batch_size * sizeof(int64_t);
    uint64_t comm_volume_2 = cache_sz * 2;
    uint64_t comm_volume_3 = sizeof(int64_t) * host_block_table_size;
    uint64_t comm_volume_4 = batch_size * sizeof(int64_t);
    uint64_t comm_volume_5 = total_tokens_size * sizeof(int64_t);
    total_comm_volume = comm_volume_1 + comm_volume_2 + comm_volume_3 + comm_volume_4 + comm_volume_5;
    printf("Comm volume breakdown:\n");
    printf("Comm volume 1: %lu bytes\n", comm_volume_1);
    printf("Comm volume 2: %lu bytes\n", comm_volume_2);
    printf("Comm volume 3: %lu bytes\n", comm_volume_3);
    printf("Comm volume 4: %lu bytes\n", comm_volume_4);
    printf("Comm volume 5: %lu bytes\n", comm_volume_5);
    printf("Total comm volume: %lu bytes\n", total_comm_volume);
    printf("Send time: %lf-%lf,%lf ms\n", timer.getElapsedTime(start), timer.getElapsedTime(stop), comm_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free all resources
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_context_output));
    CUDA_CHECK(cudaFree(d_allocated_block_cnt));
    CUDA_CHECK(cudaFree(serialized_tokens));
    //delete[] input_tokens_sizes;

    //delete model;
    // CUDA_CHECK(cudaStreamDestroy(stream));

    return comm_time;
  }

// models with templates
  st::model::Gpt<T> *model;
  T* *d_k_cache, *d_v_cache;
  std::vector<TokenList> input_tokens_batched;
  std::vector<int64_t> context_output_batched;
  std::vector<int> input_lens;
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
  std::vector<TokenList> input_tokens_batched;//syj: get the input tokens

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
    if (parallel_param.tensor_para_size > 1) {
      model->init_communicator(tp_id, pp_id);
    }
    model->initDummyWeight();
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  void PreProcess() {
    timer.startTiming();
    int peer = rank - world_size / 2;
    // init the datasize
    const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    size_t whole_cache_sz = sizeof(T) * max_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;
    
    int64_t *context_output_size;
    int64_t *d_k_cache_size;
    int64_t *d_v_cache_size;
    int64_t *block_table_size;
    int64_t *input_tokens_batched_size;
    int64_t *input_tokens_sizes; // suppose the max batch =20

    //cudamalloc this var
    CUDA_CHECK(cudaMalloc((void**)&context_output_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_k_cache_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_v_cache_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&block_table_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&input_tokens_batched_size, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc((void**)&input_tokens_sizes, 20 * sizeof(int64_t)));

    timer.stopTiming("prepare for receive data");
    timer.startTiming();
    // Receive data sizes first
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ncclGroupStart();
    NCCL_CHECK(ncclRecv(context_output_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv(d_k_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv(d_v_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv(block_table_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv(input_tokens_batched_size, 1, ncclInt64, peer, inter_stage_comm, stream)); // &input_tokens_batched_size
    NCCL_CHECK(ncclRecv(input_tokens_sizes, 20 , ncclInt64, peer, inter_stage_comm, stream)); // ? 
    ncclGroupEnd();
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.stopTiming("nccl receive size");
    timer.startTiming();
    // 在主机内存中定义相应的变量
    int64_t host_context_output_size;
    int64_t host_d_k_cache_size;
    int64_t host_d_v_cache_size;
    int64_t host_block_table_size;
    int64_t host_input_tokens_batched_size;
    int64_t host_input_tokens_sizes[20];

    // 从 GPU 内存拷贝数据到主机内存
    CUDA_CHECK(cudaMemcpy(&host_context_output_size, context_output_size, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_d_k_cache_size, d_k_cache_size, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_d_v_cache_size, d_v_cache_size, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_block_table_size, block_table_size, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_input_tokens_batched_size, input_tokens_batched_size, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_input_tokens_sizes, input_tokens_sizes, 20 * sizeof(int64_t), cudaMemcpyDeviceToHost));

    batch_size = host_context_output_size; // update the batchsize

    output_tokens_batched.resize(batch_size);
    next_iter_input_tokens_batched.resize(batch_size);
    first_token_indexes.resize(batch_size);
    input_tokens_size.resize(batch_size);

    // Allocate memory based on received sizes
    int64_t *context_output_tokens;
    int64_t* d_allocated_block_cnt;
    CUDA_CHECK(cudaMallocManaged(&context_output_tokens, sizeof(int64_t) * host_context_output_size));
    CUDA_CHECK(cudaMalloc(&d_k_cache, whole_cache_sz)); 
    CUDA_CHECK(cudaMalloc(&d_v_cache, whole_cache_sz));
    CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * max_batch_size * pagedattn_param.max_num_block_per_req));
    CUDA_CHECK(cudaMalloc(&d_allocated_block_cnt, sizeof(int64_t) * batch_size));
    int64_t* serialized_tokens;
    CUDA_CHECK(cudaMallocManaged(&serialized_tokens, sizeof(int64_t) * host_input_tokens_batched_size));
    CUDA_CHECK(cudaDeviceSynchronize());

    num_allocated_blocks = 0;
    allocated_block_cnt = new int64_t[batch_size];

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    timer.stopTiming("prepare to receive actual data");
    timer.startTiming();
    // Receive actual data
    ncclGroupStart();
    NCCL_CHECK(ncclRecv(context_output_tokens, host_context_output_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_k_cache, host_d_k_cache_size, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_v_cache, host_d_k_cache_size, ncclChar, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_block_table, host_block_table_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv((void *)d_allocated_block_cnt, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    NCCL_CHECK(ncclRecv(serialized_tokens, host_input_tokens_batched_size, ncclInt64, peer, inter_stage_comm, stream));
    ncclGroupEnd();
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.stopTiming("receive actual data");
    //Deserialize input_tokens_batched

    // comment for simplicity
    // input_tokens_batched.clear(); //clear history input_tokens
    // size_t offset = 0;
    // for (int i = 0; i < batch_size; ++i) {
    //     TokenList token_list;
    //     size_t token_list_size = host_input_tokens_sizes[i];
    //     token_list.resize(token_list_size);
    //     //memcpy(token_list.data(), serialized_tokens + offset, sizeof(int64_t) * token_list_size);
    //     CUDA_CHECK(cudaMemcpy(token_list.data(), serialized_tokens + offset, sizeof(int64_t) * token_list.size(), cudaMemcpyDeviceToHost));
    //     offset += token_list_size;
    //     input_tokens_batched.push_back(token_list);
    // }
    // comment out ending

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpyAsync(allocated_block_cnt, d_allocated_block_cnt, sizeof(int64_t) * batch_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventElapsedTime(&comm_time, start, stop));
    printf("recv time: %f-%f,%f ms\n", timer.getElapsedTime(start), timer.getElapsedTime(stop), comm_time);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // comment for debugging
    // for (uint i = 0; i < batch_size; i++)
    //   output_tokens_batched[i].push_back(context_output_tokens[i]);
    // for (uint i = 0; i < batch_size; i++)
    //   next_iter_input_tokens_batched[i].push_back(context_output_tokens[i]);
    // for (uint i = 0; i < batch_size; i++) {
    //     first_token_indexes[i] = input_tokens_batched[i].size();
    // }



    // cudaMalloc((void**)&context_output_size, sizeof(int64_t));
    // cudaMalloc((void**)&d_k_cache_size, sizeof(int64_t));
    // cudaMalloc((void**)&d_v_cache_size, sizeof(int64_t));
    // cudaMalloc((void**)&block_table_size, sizeof(int64_t));
    // cudaMalloc((void**)&input_tokens_batched_size, sizeof(int64_t));
    // cudaMalloc((void**)&input_tokens_sizes, 20 * sizeof(int64_t));

    CUDA_CHECK(cudaFree(context_output_size));
    CUDA_CHECK(cudaFree(d_k_cache_size));
    CUDA_CHECK(cudaFree(d_v_cache_size));
    CUDA_CHECK(cudaFree(block_table_size));
    CUDA_CHECK(cudaFree(input_tokens_batched_size));
    CUDA_CHECK(cudaFree(input_tokens_sizes));
    

    CUDA_CHECK(cudaFree(context_output_tokens));
    CUDA_CHECK(cudaFree(d_allocated_block_cnt));
    CUDA_CHECK(cudaFree(serialized_tokens));


    // int peer = rank - world_size / 2;

    // // 初始化数据大小信息
    // const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
    // size_t whole_cache_sz = sizeof(T) * max_block_size * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim;

    // // 在主机内存中定义相应的变量用于接收大小信息
    // int64_t host_context_output_size;
    // int64_t host_d_k_cache_size;
    // int64_t host_d_v_cache_size;
    // int64_t host_block_table_size;
    // int64_t host_input_tokens_batched_size;
    // int64_t host_input_tokens_sizes[20];

    // // 接收每部分数据的大小
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // ncclGroupStart();
    // NCCL_CHECK(ncclRecv(&host_context_output_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(&host_d_k_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(&host_d_v_cache_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(&host_block_table_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(&host_input_tokens_batched_size, 1, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(host_input_tokens_sizes, 20, ncclInt64, peer, inter_stage_comm, stream));
    // ncclGroupEnd();
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // // 更新 batch_size
    // batch_size = host_context_output_size;

    // // 分配内存基于接收到的大小信息
    // int64_t* context_output_tokens;
    // int64_t* d_allocated_block_cnt;
    // CUDA_CHECK(cudaMallocManaged(&context_output_tokens, sizeof(int64_t) * host_context_output_size));
    // CUDA_CHECK(cudaMalloc(&d_k_cache, whole_cache_sz)); 
    // CUDA_CHECK(cudaMalloc(&d_v_cache, whole_cache_sz));
    // CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * max_batch_size * pagedattn_param.max_num_block_per_req));
    // CUDA_CHECK(cudaMalloc(&d_allocated_block_cnt, sizeof(int64_t) * batch_size));
    // int64_t* serialized_tokens;
    // CUDA_CHECK(cudaMallocManaged(&serialized_tokens, sizeof(int64_t) * host_input_tokens_batched_size));
    // CUDA_CHECK(cudaDeviceSynchronize());

    // // 接收实际数据
    // ncclGroupStart();
    // NCCL_CHECK(ncclRecv(context_output_tokens, host_context_output_size, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv((void *)d_k_cache, host_d_k_cache_size, ncclChar, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv((void *)d_v_cache, host_d_v_cache_size, ncclChar, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv((void *)d_block_table, host_block_table_size, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv((void *)d_allocated_block_cnt, batch_size, ncclInt64, peer, inter_stage_comm, stream));
    // NCCL_CHECK(ncclRecv(serialized_tokens, host_input_tokens_batched_size, ncclInt64, peer, inter_stage_comm, stream));
    // ncclGroupEnd();
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // // 反序列化 input_tokens_batched
    // input_tokens_batched.clear(); // 清空历史输入 tokens
    // size_t offset = 0;
    // for (int i = 0; i < batch_size; ++i) {
    //     TokenList token_list;
    //     size_t token_list_size = host_input_tokens_sizes[i];
    //     token_list.resize(token_list_size);
    //     memcpy(token_list.data(), serialized_tokens + offset, sizeof(int64_t) * token_list_size);
    //     offset += token_list_size;
    //     input_tokens_batched.push_back(token_list);
    // }

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, stream);

    // // 更新 output_tokens_batched 和 next_iter_input_tokens_batched
    // for (uint i = 0; i < batch_size; i++) {
    //     output_tokens_batched[i].push_back(context_output_tokens[i]);
    //     next_iter_input_tokens_batched[i].push_back(context_output_tokens[i]);
    //     first_token_indexes[i] = input_tokens_batched[i].size();
    // }

    // // 记录通信时间
    // cudaEventRecord(stop, stream);
    // cudaEventSynchronize(stop);
    // CUDA_CHECK(cudaMemcpyAsync(allocated_block_cnt, d_allocated_block_cnt, sizeof(int64_t) * batch_size, cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // float comm_time;
    // cudaEventElapsedTime(&comm_time, start, stop);
    // printf("recv time: %f ms\n", comm_time);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // // 释放 GPU 内存
    // CUDA_CHECK(cudaFree(context_output_tokens));
    // CUDA_CHECK(cudaFree(d_allocated_block_cnt));
    // CUDA_CHECK(cudaFree(serialized_tokens));
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
      // // Allocate new blocks if needed
      // for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
      //   // todo: check if output_tokens_batched is initialized
      //   int64_t block_needed = ROUND_UP(input_tokens_size[request_id[i]]
      //               + output_tokens_batched[request_id[i]].size(), pagedattn_param.block_size) / pagedattn_param.block_size;
      //   while (allocated_block_cnt[request_id[i]] < block_needed) {
      //     int64_t new_block = allocateNewBlock();
      //     CUDA_CHECK(cudaMemcpy(d_block_table + i*pagedattn_param.max_num_block_per_req + allocated_block_cnt[request_id[i]], &new_block, sizeof(int64_t), cudaMemcpyHostToDevice));
      //     allocated_block_cnt[request_id[i]] += 1;
      //   }
      // }
      // // sync_check_cuda_error();
      // CUDA_CHECK(cudaDeviceSynchronize());
      // auto cur_iter_output_tokens = model->forward(
      //   next_iter_input_tokens_batched,
      //   first_token_indexes,

      //   d_k_cache,
      //   d_v_cache,
      //   d_block_table	
      // );
      // CUDA_CHECK(cudaDeviceSynchronize());
      // // Decode generated token and put them into output_tokens_batched
      // // Prepare input for the next round
      // int64_t ptr = 0;
      // int64_t new_notdone_input_cnt = cur_notdone_input_cnt;
      // next_iter_input_tokens_batched.clear();
      // first_token_indexes.clear();
      // for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
      //   int64_t result_token = cur_iter_output_tokens[i];
      //   output_tokens_batched[request_id[i]].push_back(result_token);
      //   if (result_token == end_token) {
      //     // The generation of this request is done
      //     --new_notdone_input_cnt;
      //   } else {
      //     next_iter_input_tokens_batched.push_back(std::vector<int64_t>{result_token});
      //     first_token_indexes.push_back(input_tokens_size[request_id[i]] + output_tokens_batched[request_id[i]].size() - 1);
      //     // Copy k/v cache to the right place if necessary (can be optimized in the future)
      //     if (i != ptr) {
      //       request_id[ptr] = request_id[i];
      //       CUDA_CHECK(cudaMemcpyAsync(
      //         d_block_table + ptr*pagedattn_param.max_num_block_per_req,
      //         d_block_table + i*pagedattn_param.max_num_block_per_req,
      //         sizeof(int64_t) * pagedattn_param.max_num_block_per_req,
      //         cudaMemcpyDeviceToDevice
      //       ));
      //       // sync_check_cuda_error();
      //       CUDA_CHECK(cudaDeviceSynchronize());
      //     }
      //     ptr += 1;
      //   }
      // }
      // cur_notdone_input_cnt = new_notdone_input_cnt;
      // sleep 25ms
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    timer.stopTiming("decode forward computation");
    // timer.printTimes();
    delete[] request_id;
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_block_table));
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