CUDA_HOME?=$(CODNA_PREFIX)
$(info $(CUDA_HOME))
BUILDDIR:=./build
NCCL_HOME:=./nccl
NCCL_BUILD_DIR?=$(NCCL_HOME)/build
NCCL_INC:=$(NCCL_HOME)/src/include
NCCL_HEADER:=$(NCCL_BUILD_DIR)/include
FL_SO:=libnccl-net-fuselink.so
NVCC:=$(CUDA_HOME)/bin/nvcc
CXX:=/usr/bin/g++
INC:=$(CUDA_CFLAGS) -I$(NCCL_INC) -I$(NCCL_HEADER) -I$(CONDA_PREFIX)/include -Isrc/
LIB_PATH:=-L/usr/lib/x86_64-linux-gnu -L$(NCCL_BUILD_DIR)/lib $(CUDA_LDFLAGS) -L$(CONDA_PREFIX)/lib

CFLAGS:=-fPIC -O2 -Wno-deprecated-gpu-targets
TEST_LIBS:=-lgtest -lgtest_main -lpthread -libverbs -lrt -lcuda -lcudart

.PHONY: clean all test

monitor: src/monitor_main.cpp src/monitor.cpp
	g++ $(INC) -o monitor src/monitor_main.cpp src/monitor.cpp -lpthread -libverbs -lrt

monitor_test: src/monitor_test.cpp src/monitor.cpp
	g++ $(INC) -O0 -g -o monitor_test src/monitor_test.cpp src/monitor.cpp $(LIB_PATH) $(TEST_LIBS) 

monitor_client_test: src/monitor_client_test.cpp src/monitor.cpp
	g++ $(INC) -O0 -g -o monitor_client_test src/monitor_client_test.cpp src/monitor.cpp $(LIB_PATH) $(TEST_LIBS) 

cumem_test: tests/cumem_test.cu src/cumem.cu nccl/src/misc/cudawrap.cc
	$(NVCC) $(INC) -O2 -c -Xcompiler -fPIC -o cumem_test.o tests/cumem_test.cu
	$(NVCC) $(INC) -O2 -c -Xcompiler -fPIC -o cudawrap.o nccl/src/misc/cudawrap.cc
	$(NVCC) $(INC) -O2 -c -Xcompiler -fPIC -o cumem.o src/cumem.cu
	g++ $(INC) $(LIB_PATH) -O0 -g -o cumem_test cumem_test.o cudawrap.o cumem.o  $(TEST_LIBS) 

test: monitor_test monitor_client_test
	./monitor_test
	./monitor_client_test

FLSRC:=src/net_multi_nic.cc \
		src/fuselink.cc \
        nccl/src/misc/ibvwrap.cc \
		nccl/src/misc/socket.cc \
		nccl/src/misc/cudawrap.cc \
		nccl/src/misc/utils.cc \
		nccl/src/misc/ibvsymbols.cc \
		nccl/src/misc/param.cc \
		nccl/src/debug.cc \

CUSRC:=src/cumem.cu

FLOBJS:=$(FLSRC:.cc=.o)
CUOBJS:=$(CUSRC:.cu=.o)

fl: $(BUILDDIR)/lib/$(FL_SO)
all: fl monitor test

$(BUILDDIR)/lib/$(FL_SO): $(FLOBJS) $(CUOBJS)
	mkdir -p $(BUILDDIR)/lib
	$(NVCC) $(LIB_PATH) -shared -Xcompiler $(CFLAGS) -libverbs -lnccl -lcuda -o $(BUILDDIR)/lib/$(FL_SO) $(FLOBJS) $(CUOBJS)

%.o: %.cc
	mkdir -p build/src
	$(NVCC) $(INC) -O2 -c -Xcompiler -fPIC -o $@ $<

%.o: %.cu
	mkdir -p build/src
	$(NVCC) $(INC) -O2 -c -Xcompiler -fPIC -o $@ $<

clean:
	rm -rf build
	rm -f src/*.o
	rm -f monitor_test monitor_client_test