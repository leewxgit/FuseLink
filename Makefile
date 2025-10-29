CUDA_HOME?=$(CODNA_PREFIX)
$(info $(CUDA_HOME))
BUILDDIR:=./build
NCCL_HOME:=./nccl
NCCL_BUILD_DIR?=$(NCCL_HOME)/build
NCCL_INC:=$(NCCL_HOME)/src/include
NCCL_HEADER:=$(NCCL_BUILD_DIR)/include
FL_SO:=libnccl-net-unet.so
NVCC:=$(CUDA_HOME)/bin/nvcc
CXX:=/usr/bin/g++
INC:=$(CUDA_CFLAGS) -I$(NCCL_INC) -I$(NCCL_HEADER) -I$(CONDA_PREFIX)/include -Isrc/
LIB_PATH:=-L/usr/lib/x86_64-linux-gnu -L$(NCCL_BUILD_DIR)/lib $(CUDA_LDFLAGS) -L$(CONDA_PREFIX)/lib

CFLAGS:=-fPIC -O2 -Wno-deprecated-gpu-targets

.PHONY: clean all test

monitor: src/monitor_main.cpp src/monitor.cpp
	g++ $(INC) -o monitor src/monitor_main.cpp src/monitor.cpp -lpthread -libverbs -lrt

FLSRC:=src/plugin_unet.cc \
		src/unet.cc \
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
all: fl monitor

$(BUILDDIR)/lib/$(FL_SO): $(FLOBJS) $(CUOBJS)
	mkdir -p $(BUILDDIR)/lib
	$(NVCC) $(LIB_PATH) -shared -Xcompiler $(CFLAGS) -libverbs -lnccl -lcuda -o $(BUILDDIR)/lib/$(FL_SO) $(FLOBJS) $(CUOBJS)

%.o: %.cc
	mkdir -p build/src
	$(NVCC) $(INC) -O2 -c -Xcompiler $(CFLAGS) -o $@ $<

%.o: %.cu
	mkdir -p build/src
	$(NVCC) $(INC) -O2 -c -Xcompiler $(CFLAGS) -o $@ $<

clean:
	rm -rf build
	rm -f src/*.o