cmake_minimum_required(VERSION 3.18)
project(FuseLinkServeTest LANGUAGES CXX CUDA)

# set CUDA compiler 
if (DEFINED ENV{CONDA_PREFIX})
  # use conda environment
  link_directories($ENV{CONDA_PREFIX}/lib)
  include_directories($ENV{CONDA_PREFIX}/include)
  set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
endif()

if (DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CUDAToolkit_ROOT $ENV{CONDA_PREFIX})
  set(CMAKE_CUDA_COMPILER $ENV{CONDA_PREFIX}/bin/nvcc)
endif()
  
find_package(CUDAToolkit REQUIRED)

# Set up C++ standard
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# download SwiftTransformer to source dir
if (NOT EXISTS ${ST_ROOT_DIR})
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/SwiftTransformer)
        message("Downloading SwiftTransformer")
        execute_process(COMMAND git clone --recursive https://github.com/LLMServe/SwiftTransformer.git ${CMAKE_CURRENT_SOURCE_DIR}/SwiftTransformer )
    endif()
    set(ST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SwiftTransformer)
    set(SwiftTransformer_INCLUDE_DIRS ${ST_ROOT_DIR}/src/csrc)
endif()


# set python environment
set(PYTHON_PATH "python" CACHE STRING "Python path")


# Add Python into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
print(sysconfig.get_paths()['include']);"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Python config Error.")
endif()

# add libtorch environment
execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
print(os.path.dirname(torch.__file__), end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)

execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Set up MPI and NCCL for multi-GPU communication
message("Building with MPI and NCCL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# show CMAKE_MODULE_PATH
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
set(MKL_MPI "openmpi")
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

#require boost
# find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PY_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${NCCL_INCLUDE_DIRS})
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
include_directories(${SwiftTransformer_INCLUDE_DIRS})

add_subdirectory(${ST_ROOT_DIR}/src/csrc/kernel)
add_subdirectory(${ST_ROOT_DIR}/src/csrc/layer)
add_subdirectory(${ST_ROOT_DIR}/src/csrc/model)
add_subdirectory(${ST_ROOT_DIR}/src/csrc/util)
add_subdirectory(${ST_ROOT_DIR}/src/examples)

# nlohmann_json Preparation - Code block copied from
#   https://github.com/nlohmann/json#cmake
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# fetch latest argparse
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

ADD_EXECUTABLE(FuseLinkServeTest main.cc)
TARGET_INCLUDE_DIRECTORIES(FuseLinkServeTest  PUBLIC ${SwiftTransformer_INCLUDE_DIRS})

target_link_libraries(FuseLinkServeTest  PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(FuseLinkServeTest  PUBLIC 
                        model_gpt
                        model_opt
                        model_llama2
                        model_gpt2
                        py_nccl_utils
                        py_swapping
                        py_block_migration
                    )
target_link_libraries(FuseLinkServeTest  PUBLIC ${MPI_LIBRARIES})
target_link_libraries(FuseLinkServeTest  PUBLIC ${NCCL_LIBRARIES})
target_link_libraries(FuseLinkServeTest  PUBLIC CUDA::cudart)


ADD_EXECUTABLE(FuseLinkOnlineServeTest online_main.cc)
TARGET_INCLUDE_DIRECTORIES(FuseLinkOnlineServeTest  PUBLIC ${SwiftTransformer_INCLUDE_DIRS})

target_link_libraries(FuseLinkOnlineServeTest  PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(FuseLinkOnlineServeTest  PUBLIC 
                        model_gpt
                        model_opt
                        model_llama2
                        model_gpt2
                        py_nccl_utils
                        py_swapping
                        py_block_migration
                    )
target_link_libraries(FuseLinkOnlineServeTest  PUBLIC ${MPI_LIBRARIES})
target_link_libraries(FuseLinkOnlineServeTest  PUBLIC ${NCCL_LIBRARIES})
target_link_libraries(FuseLinkOnlineServeTest  PUBLIC CUDA::cudart)



target_link_libraries(benchmark_all_input_same ${TORCH_LIBRARIES})
target_link_libraries(benchmark_all_input_same
  model_gpt
  model_opt
  model_llama2
  model_gpt2
  py_nccl_utils
  py_swapping
  py_block_migration
  CUDA::cudart
)

target_link_libraries(run_gpt ${TORCH_LIBRARIES})
target_link_libraries(run_gpt
  model_gpt
  model_opt
  model_llama2
  model_gpt2
  py_nccl_utils
  py_swapping
  py_block_migration
  CUDA::cudart
)