#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <cuda.h>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>
// include some headers from the SwiftTransformer project
#include "model.hh"
// build program options without boost


typedef __half half;

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
        params.input_len = std::stoll(options.at("--input_len"));
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

int main(int argc, char** argv) {

  // Parse options
  TestParams params;
  auto param_map = parse_options_to_map(argc, argv);
  set_params(param_map, params);

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
  cudaSetDevice(local_rank);

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

  // set up input tokens in test params
  TokenList input_tokens_crafted = input_tokens;
  if (input_tokens_crafted.size() > params.input_len) {
    input_tokens_crafted.resize(params.input_len);
  } else {
    while (input_tokens_crafted.size() < params.input_len) {
      input_tokens_crafted.push_back(input_tokens_crafted.back());
    }
  }
  std::vector<TokenList> input_tokens_batched;
  for (int i = 0; i < params.batch_size; i++) {
    input_tokens_batched.push_back(input_tokens_crafted);
  }


  // set up inter_stage_comm, rank, world_size in test params
  params.inter_stage_comm = nccl_world_comm;
  params.rank = world_rank;
  params.world_size = world_size;

  float send_time, recv_time;
  double t1, t2;
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  sleep(30);
  if (params.precision == "fp16") {
    if (world_rank < world_size / 2) {
      // world_rank < world_size / 2
      // context
      TestContextStageModel<half> model;
      model.LoadTestParams(params);
      model.PreProcessInit();
      model.PreProcess(input_tokens_batched);
      model.Forward();
      send_time = model.PostProcess();
    } else {
      // decoding
      TestDecodingStageModel<half> model;
      model.LoadTestParams(params);
      model.PreProcessInit();
      model.PreProcess(input_tokens_batched);
      model.Forward();
      recv_time = model.PostProcess();
    }
  } else if (params.precision == "fp32") {
    if (world_rank < world_size / 2) {
      // world_size / 2
      // context
      TestContextStageModel<float> model;
      model.LoadTestParams(params);
      model.PreProcessInit();
      model.PreProcess(input_tokens_batched);
      model.Forward();
      send_time = model.PostProcess();
    } else {
      // decoding
      TestDecodingStageModel<float> model;
      model.LoadTestParams(params);
      model.PreProcess(input_tokens_batched);
      model.Forward();
      recv_time = model.PostProcess();
    }
  } else {
    std::cerr << "Invalid precision" << std::endl;
    exit(1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();

  // print profiling results
  if (world_rank == 0){
    printf("Elapsed time: %f ms\n", (t2 - t1)*1000);
  }
  
  if (world_rank < world_size / 2){
    MPI_Send(&send_time, 1, MPI_DOUBLE, world_rank + world_size / 2, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&send_time, 1, MPI_DOUBLE, world_rank - world_size / 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Communication time for peer %d: %f ms\n", world_size % world_rank, send_time + recv_time);
    printf("Ratio for peer %d: %.3lf\n", world_size % world_rank, (send_time + recv_time) / ((t2 - t1)*1000));
  }

  MPI_Finalize();

}