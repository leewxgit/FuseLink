#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <cuda.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>
#include <random>
// include some headers from the SwiftTransformer project
#include "online_model.hh"
// build program options without boost

typedef __half half;

TestParams params;

TokenList input_tokens = {3762, 6, 80, 6, 130, 6, 237, 6, 292, 6, 411, 6, 707, 6, 799, 6, 1117, 6, 2724, 6, 19353, 6, 11971, 6, 30361, 6, 31925, 6, 23843, 6, 32382, 6, 37100, 6, 34851, 6, 40126, 6, 10328, 6, 10328, 12, 1264, 6, 10328, 12, 7109, 6, 10328, 12, 9983, 6, 10328, 12, 10231, 6, 10328, 12, 9579, 6, 10328, 12, 13664, 6, 10328, 12, 17723, 6, 10328, 12, 19491, 6, 10328, 12, 22255, 6, 16984, 6, 16984, 12, 1264, 6, 16984, 12, 7109, 6, 16984, 12, 9983, 6, 16984, 12, 10231, 6, 16984, 12, 9579, 6, 16984, 12, 13664, 6, 16984, 12, 17723, 6, 16984, 12, 19491, 6, 16984, 12, 22255, 6, 24503, 6, 24503, 12, 1264, 6, 24503, 12, 7109, 6, 24503, 12, 9983, 6, 24503, 12, 10231, 6, 24503, 12, 9579, 6, 24503, 12, 13664, 6, 24503, 12, 17723, 6, 24503, 12, 19491, 6, 24503, 12, 22255, 6, 14865, 6, 14865, 12, 1264, 6, 14865, 12, 7109, 6, 14865, 12, 9983, 6, 14865, 12, 10231, 6, 14865, 12, 9579, 6, 14865, 12, 13664, 6, 14865, 12, 17723, 6, 14865, 12, 19491, 6, 14865, 12, 22255, 6, 33910, 6, 33910, 12, 1264, 6, 33910, 12, 7109, 6, 33910, 12, 9983, 6, 33910, 12, 10231, 6, 33910, 12, 9579, 6, 33910, 12, 13664, 6, 33910, 12, 17723, 6, 33910, 12, 19491, 6, 33910, 12, 22255, 6, 39676, 6, 39676, 12, 1264, 6, 39676, 12, 7109, 6, 39676, 12, 9983, 6, 39676, 12, 10231, 6, 39676, 12, 9579, 6, 39676, 12, 13664, 6, 39676, 12, 17723, 6, 39676, 12, 19491, 6, 39676, 12, 22255, 6, 42991, 6, 42991, 12, 1264, 6, 42991, 12, 7109, 6, 42991, 12, 9983, 6, 42991, 12, 10231, 6, 42991, 12, 9579, 6, 42991, 12, 13664, 6, 42991, 12, 17723, 6, 42991, 12, 19491, 6, 42991, 12, 22255, 6, 33035, 6, 33035, 12, 1264, 6, 33035, 12, 7109, 6, 33035, 12, 9983, 6, 33035, 12, 10231, 6, 33035, 12, 9579, 6, 33035, 12, 13664, 6, 33035, 12, 17723, 6, 33035, 12, 19491, 6, 33035, 12, 22255, 6, 65, 6317, 6, 65, 6317, 65, 6, 65, 6317, 80, 6, 65, 6317, 130, 6, 65, 6317, 237, 6, 65, 6317, 292, 6, 65, 6317, 411, 6, 65, 6317, 707, 6, 65, 6317, 799, 6, 65, 6317, 1117, 6, 65, 6317, 2724, 6, 65, 6317, 19353, 6, 65, 6317, 11971, 6, 65, 6317, 30361, 6, 65, 6317, 31925, 6, 65, 6317, 23843, 6, 65, 6317, 32382, 6, 65, 6317, 37100, 6, 65, 6317, 34851, 6, 65, 6317, 40126, 6, 65, 6317, 10328, 6, 65, 6317, 10328, 12, 1264, 6, 65, 6317, 10328, 12, 7109, 6, 65, 6317, 10328, 12, 9983, 6, 65, 6317, 10328, 12, 10231, 6, 65, 6317, 10328, 12, 9579, 6, 65, 6317, 10328, 12, 13664, 6, 65, 6317, 10328, 12, 17723, 6, 65, 6317, 10328, 12, 19491, 6, 65, 6317, 10328, 12, 22255, 6, 65, 6317, 16984, 6, 65, 6317, 16984, 12, 1264, 6, 65, 6317, 16984, 12, 7109, 6, 65, 6317, 16984, 12, 9983, 6, 65, 6317, 16984, 12, 10231, 6, 65, 6317, 16984, 12, 9579, 6, 65, 6317, 16984, 12, 13664, 6, 65, 6317, 16984, 12, 17723, 6, 65, 6317, 16984, 12, 19491, 6, 65, 6317, 16984, 12, 22255, 6, 65, 6317, 24503, 6, 65, 6317, 24503, 12, 1264, 6, 65, 6317, 24503, 12, 7109, 6, 65, 6317, 24503, 12, 9983, 6, 65, 6317, 24503, 12, 10231, 6, 65, 6317, 24503, 12, 9579, 6, 65, 6317, 24503, 12, 13664, 6, 65, 6317, 24503, 12, 17723, 6, 65, 6317, 24503, 12, 19491, 6, 65, 6317, 24503, 12, 22255, 6, 65, 6317, 14865, 6, 65, 6317, 14865, 12, 1264, 6, 65, 6317, 14865, 12, 7109, 6, 65, 6317, 14865, 12, 9983, 6, 65, 6317, 14865, 12, 10231, 6, 65, 6317, 14865, 12, 9579, 6, 65, 6317, 14865, 12, 13664, 6, 65, 6317, 14865, 12, 17723, 6, 65, 6317, 14865, 12, 19491, 6, 65, 6317, 14865, 12, 22255, 6, 65, 6317, 33910, 6, 65, 6317, 33910, 12, 1264, 6, 65, 6317, 33910, 12, 7109, 6, 65, 6317, 33910, 12, 9983, 6, 65, 6317, 33910, 12, 10231, 6, 65, 6317, 33910, 12, 9579, 6, 65, 6317, 33910, 12, 13664, 6, 65, 6317, 33910, 12, 17723, 6, 65, 6317, 33910, 12, 19491, 6, 65, 6317, 33910, 12, 22255, 6, 65, 6317, 39676, 6, 65, 6317, 39676, 12, 1264, 6, 65, 6317, 39676, 12, 7109, 6, 65, 6317, 39676, 12, 9983, 6, 65, 6317, 39676, 12, 10231, 6, 65, 6317, 39676, 12, 9579, 6, 65, 6317, 39676, 12, 13664, 6, 65, 6317, 39676, 12, 17723, 6, 65, 6317, 39676, 12, 19491, 6, 65, 6317, 39676, 12, 22255, 6, 65, 6317, 42991, 6, 65, 6317, 42991, 12, 1264, 6, 65, 6317, 42991, 12, 7109, 6, 65, 6317, 42991, 12, 9983, 6, 65, 6317, 42991, 12, 10231, 6, 65, 6317, 42991, 12, 9579, 6, 65, 6317, 42991, 12, 13664, 6, 65, 6317, 42991, 12, 17723, 6, 65, 6317, 42991, 12, 19491, 6, 65, 6317, 42991, 12, 22255, 6, 65, 6317, 33035, 6, 65, 6317, 33035, 12, 1264, 6, 65, 6317, 33035, 12, 7109, 6, 65, 6317, 33035, 12, 9983, 6, 65, 6317, 33035, 12, 10231, 6, 65, 6317, 33035, 12, 9579, 6, 65, 6317, 33035, 12, 13664, 6, 65, 6317, 33035, 12, 17723, 6, 65, 6317, 33035, 12, 19491, 6, 65, 6317, 33035, 12, 22255, 6, 80, 6317, 4};

// std::string options[] = {
//   "--model_path",
//   "--model_name",
//   "--vocab_json_path",
//   "--precision",
//   "--input_len",
//   "--batch_size",
//   "--block_size",
//   "--num_decoding_step",
//   "--tp_size"
// };
std::string tracefile;
std::map<std::string, std::string> parse_options_to_map(int argc, char** argv) {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            options[argv[i]] = argv[i + 1];
        } else {
            std::cerr << "Missing value for option " << argv[i] << std::endl;
            exit(1);
        }
    }
    return options;
}

// Parse options

void set_params(const std::map<std::string, std::string>& options, TestParams& params) {
    if (options.find("--model_path") != options.end()) {
        params.model_path = options.at("--model_path");
    } else {
        std::cerr << "Missing required argument: --model_path" << std::endl;
        exit(1);
    }
    if (options.find("--tracepath") != options.end()) {
        tracefile = options.at("--tracepath");
    } else {
        std::cerr << "Missing required argument: --tracepath" << std::endl;
        exit(1);
    }

    if (options.find("--model_name") != options.end()) {
        params.model_name = options.at("--model_name");
    } else {
        std::cerr << "Missing required argument: --model_name" << std::endl;
        exit(1);
    }

    if (options.find("--vocab_json_path") != options.end()) {
        params.vocab_json_path = options.at("--vocab_json_path");
    } else {
        params.vocab_json_path = "";
    }

    if (options.find("--precision") != options.end()) {
        params.precision = options.at("--precision");
    } else {
        std::cerr << "Missing required argument: --precision" << std::endl;
        exit(1);
    }

    if (options.find("--input_len") != options.end()) {
        params.input_len = std::stoll(options.at("--input_len")); //it should be max inputlen
    } else {
        std::cerr << "Missing required argument: --input_len" << std::endl;
        exit(1);
    }

    if (options.find("--batch_size") != options.end()) {
        params.batch_size = std::stoll(options.at("--batch_size"));
    } else {
        std::cerr << "Missing required argument: --batch_size" << std::endl;
        exit(1);
    }

    if (options.find("--block_size") != options.end()) {
        params.block_size = std::stoll(options.at("--block_size"));
    } else {
        std::cerr << "Missing required argument: --block_size" << std::endl;
        exit(1);
    }

    if (options.find("--num_decoding_step") != options.end()) {
        params.num_decoding_step = std::stoll(options.at("--num_decoding_step"));
    } else {
        std::cerr << "Missing required argument: --num_decoding_step" << std::endl;
        exit(1);
    }

    if (options.find("--tp_size") != options.end()) {
        params.tp_sz = std::stoll(options.at("--tp_size"));
    } else {
        std::cerr << "Missing required argument: --tp_size" << std::endl;
        exit(1);
    }
}

TokenList generate_random_tokenlist(int requestlen) {
    TokenList req;
    req.reserve(requestlen);

    // 设置随机数生成器，取值范围为 0 到 31999
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(0, 31999);

    // 生成随机的 tokenlist
    for (int i = 0; i < requestlen; ++i) {
        req.push_back(dis(gen));
    }

    return req;
}

void context_process(TestContextStageModel<half>& context_model) {
  int counter=0;
  while(true){
    std::unique_lock<std::mutex> lock(queueMutex);
    queue_cv.wait(lock, [] { return !request_queue.empty(); });
    // Process the request and batch them together
    std::vector<TokenList> input_tokens_batched;
    std::vector<int> total_prompt_len;
    while (!request_queue.empty() && input_tokens_batched.size() < context_model.max_batch_size) {
      Request req_info = request_queue.front();
      if(req_info.active_prompt_len > params.input_len)
        continue;
      request_queue.pop();
      input_tokens_batched.push_back(generate_random_tokenlist(req_info.active_prompt_len)); // generate dummy input
      total_prompt_len.push_back(req_info.prompt_len);
    }
    lock.unlock();
    //input_tokens_batched should be the new tokens
    // TODO: 
    //    1. need to parse the batchsize inside the models
    //    2. need to change volume according to promptlen + computation according to active prompt_len
    fprintf(stderr,"now at %d \n",counter++);
    assert(input_tokens_batched.size() <= context_model.max_batch_size);
    context_model.PreProcess(input_tokens_batched, total_prompt_len); 
    context_model.Forward();
    context_model.PostProcess();// send kv cache
  }
}

void decode_process(TestDecodingStageModel<half>& decode_model) {
  while(true){
    //TODO: add logic if done sleep for xxx ms
    //    1. add logic
    decode_model.PreProcess(); //recieve data from context process
    decode_model.Forward();
  }
}



int main(int argc, char** argv) {

  // Parse options
  auto param_map = parse_options_to_map(argc, argv);
  set_params(param_map, params);

//   sleep(60);

  MPI_Init(&argc, &argv);
  int world_rank = -1;
  int world_size = -1;
  int local_rank = -1;
  int local_size = -1;
  int pp_rank = -1;
  int tp_rank = -1;

  MPI_Comm mpi_local_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &mpi_local_comm);
  MPI_Comm_rank(mpi_local_comm, &local_rank);
  MPI_Comm_size(mpi_local_comm, &local_size);
  printf("world_rank: %d, world_size: %d, local_rank: %d, local_size: %d\n", world_rank, world_size, local_rank, local_size);

  ncclUniqueId world_id;
  if (world_rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&world_id));
  }
  MPI_Bcast(&world_id, sizeof(world_id), MPI_BYTE, 0, MPI_COMM_WORLD);

  // init nccl, tp is for tensor parallelism
  ncclComm_t nccl_world_comm;
  
  NCCL_CHECK(ncclCommInitRank(&nccl_world_comm, world_size, world_id, world_rank));
  
  // Set device on local_rank

  // set up tp_rk in test params
  params.tp_rk = local_rank;

  // set up tp_id, pp_id in test params
  ncclUniqueId tp_id, pp_id;
  ncclGetUniqueId(&pp_id);
  if (local_rank == 0) {
    ncclGetUniqueId(&tp_id);
  }
  MPI_Bcast(&tp_id, sizeof(tp_id), MPI_BYTE, 0, mpi_local_comm);
  params.tp_id = tp_id;
  params.pp_id = pp_id;

  // // set up input tokens in test params
  // TokenList input_tokens_crafted = input_tokens;
  // if (input_tokens_crafted.size() > params.input_len) {
  //   input_tokens_crafted.resize(params.input_len);
  // } else {
  //   while (input_tokens_crafted.size() < params.input_len) {
  //     input_tokens_crafted.push_back(input_tokens_crafted.back());
  //   }
  // }
  // std::vector<TokenList> input_tokens_batched;
  // for (int i = 0; i < params.batch_size; i++) {
  //   input_tokens_batched.push_back(input_tokens_crafted);
  // }


  // set up inter_stage_comm, rank, world_size in test params
  params.inter_stage_comm = nccl_world_comm;
  params.rank = world_rank;
  params.world_size = world_size;

  float send_time, recv_time;
  double t1, t2;
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  // prepare
  TestContextStageModel<half> context_model;
  TestDecodingStageModel<half> decode_model;
  sleep(60);
  if (params.precision == "fp16") {
    if (world_rank < world_size / 2) {
      //context
      context_model.LoadTestParams(params);
      context_model.PreProcessInit(); // only need once
      std::thread readerThread(readRequests, tracefile);
      std::thread contextThread(context_process,std::ref(context_model));
      readerThread.join();
      contextThread.join();
    } else {
      //decode
      decode_model.LoadTestParams(params);
      decode_model.PreProcessInit(); // only need once
      std::thread decodeThread(decode_process,std::ref(decode_model));
      decodeThread.join();
    }
  } else {
    std::cerr << "Invalid precision" << std::endl;
    exit(1);
  }
  
  // std::thread reader_thread(read_requests, "requests.txt");
  // std::thread processor_thread(process_requests);

  // reader_thread.join();
  // processor_thread.join();


  // while(true){
  //    while (true) {
  //       // TODO: read && launch request from txt file
  //       TokenList next_request = read_next_request();
  //       input_tokens_batched.clear();
  //       input_tokens_batched.push_back(next_request);

  //       if (params.precision == "fp16") {
  //           if (world_rank < world_size / 2) {
  //             // context stage
  //             // TODO: logic should change here
  //             // 1. if new request come: 
  //             //    1) it should wait until the context model finish current forward & postprocess (dont have to be async here)
  //             // 2. if no request come:
  //             //    1) it should wait until new request come
  //             std::thread readerThread(readRequests, tracefile);
  //             std::thread contextThread(context_process,context_model);
  //             readerThread.join();
  //             contextThread.join();
  //           } else {
  //             double rec_time = 0;
  //             // TODO: logic should change here: 
  //             //    1. if no new request, it should proceed decode process
  //             //    2. if there is new request, it should async recieve and add it into decode process 
  //             std::thread decodeThread(decode_process,decode_model);
  //             decodeThread.join();
  //           }
  //       } else if (params.precision == "fp32") {
  //         assert(false);
  //       } else {
  //           std::cerr << "Invalid precision" << std::endl;
  //           exit(1);
  //       }
  //       if (world_rank == 0) {
  //           printf("Elapsed time: %f ms\n", (t2 - t1) * 1000);
  //       }
  //   }
  // }
  MPI_Finalize();

}