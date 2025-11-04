#ifndef UNET_H
#define UNET_H
#include "ibvwrap.h"
#include <vector>
#include <map>
#include <string>
#include "monitor.h"

#include "nccl_net.h" // interaction with nccl net plugin

extern ncclNet_t ncclNetPlugin_v7;

#define MAX_GPU_NUM 8

#define UNET_STEPS 8

#define UPDATE_HACKING_THRESHOLD (UNET_STEPS * sizeof(void*))

struct ncclProxySubArgsForHacking {
  void* connection;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;

  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void* requests[UNET_STEPS];
  void* profilingEvents[UNET_STEPS];
  void* recvRequestsCache[UNET_STEPS];
  int recvRequestsSubCount;
}; // use by nccl, import this to hack sub->posted, sub->received, sub->transmitted, sub->done from nccl

struct UnetMemRegion {
  uint32_t sz; // 32bit is enough
  struct ibv_mr *mr[MAX_GPU_NUM + 1][MAX_NIC_NUM]; // registration on all NICs, with cpu mem on MAX_GPU_NUM index
  CUdeviceptr addr[MAX_GPU_NUM + 1]; // use this addr to enable RDMA access
  CUmemGenericAllocationHandle hdl[MAX_GPU_NUM]; // physical memory
  int nrefs; // when 0, can be freed
  pthread_mutex_t lock; // update nrefs
};

struct UnetMemHandle {
  uintptr_t start_addr;
  int dev; // mapped to which gpu, -1: cpumem
  struct UnetMemRegion *unmr;
};

typedef std::map<void*, struct UnetMemRegion*> Addr2UnetMemRegion;

enum UnetConnSetupState {
  UnetConnSetupStateInit = 0,
  UnetConnSetupStatePending = 1,
  UnetConnSetupStateReady = 2
};

class UnetConnManager {
public:
  UnetConnManager(int ndevs, int priority_dev) {
    ndevs_ = ndevs;
    priority_dev_ = priority_dev;
    memset(&switch_addr_, 0, sizeof(switch_addr_));
    switch_addr_.sin_family = AF_INET;
    switch_addr_.sin_port = htons(12345);
    switch_addr_.sin_addr.s_addr = inet_addr("192.168.1.100"); // UNET: need to change to the switch ip address
    //create client socket
    switch_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (switch_sock_fd_ < 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Unet: Failed to create switch socket");
      return ncclInternalError;
    }
  }
  int GetNdevs() {
    return ndevs_;
  }
  ~UnetConnManager() {
    // tear down all connections and monitor
    for (int i = 0; i < ndevs_; i++) {
      if (tx_send_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeSend(tx_send_comms_[i]);
      }
      if (mirror_tx_send_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeSend(mirror_tx_send_comms_[i]);
      }
      if (rx_recv_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeRecv(rx_recv_comms_[i]);
      }
      if (mirror_rx_recv_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeRecv(mirror_rx_recv_comms_[i]);
      }
    }
    // delete monitor
    delete monitor_client_;
    // close switch socket
    close(switch_sock_fd_);
  }
  void SetUpMonitor() {
    monitor_client_ = new MonitorClient();
  }

  void PushTxSendComm(void* send_comm) {
    tx_send_comms_.push_back(send_comm);
  }

  int GetTxNum() {
    return tx_send_comms_.size();
  }
  
  void PushRxRecvComm(void* recv_comm) {
    rx_recv_comms_.push_back(recv_comm);
  }
  
  int GetRxNum() {
    return rx_recv_comms_.size();
  }

  void PushMirrorTxSendComm(void* send_comm) {
    mirror_tx_send_comms_.push_back(send_comm);
  }
  
  int GetMirrorTxNum() {
    return mirror_tx_send_comms_.size();
  }
  
  void PushMirrorRxRecvComm(void* recv_comm) {
    mirror_rx_recv_comms_.push_back(recv_comm);
  }
  
  int GetMirrorRxNum() {
    return mirror_rx_recv_comms_.size();
  }

  // UNET: NIC selection related functions (monitor or manual) - begin
  bool MarkTxChannelActive(int channelId) {
    return MarkNicTxBusy(channel2dev(channelId), channelId);
  }
  bool MarkRxChannelActive(int channelId) {
    return MarkNicRxBusy(channel2dev(channelId), channelId);
  }
  void* channel2txcomm(int channelId) {
    int dev = channel2dev(channelId);
    return tx_send_comms_[dev];
  }
  void* channel2rxcomm(int channelId) {
    int dev = channel2dev(channelId);
    return rx_recv_comms_[dev];
  }
  void* bidTxChannel(int channelId, int txNicId) {
    // DEBUG: return NULL to enforce the use of comm itself
    return tx_send_comms_[txNicId];
    // DEBUG: end
    // return monitor_client_->bidTxChannel(group_id, offset);
  }
  void* refreshRxComm(int channelId, uint32_t txAvailable, int nNics, int* fuselink_offset) {
    // call monitorclient to get a new nic
    // convert txAvailable to mask

    // strategy for testing packet spraying in network
    int dev = channelId % nNics;
    *fuselink_offset = dev;
    return rx_recv_comms_[dev];

    // int group_id = channelId / NNIC_PER_GROUP;
    // int dev;
    // dev = monitor_client_->getIdleNicRx(group_id, priority_dev_, txAvailable);
    // rxchannel2dev_[channelId] = dev;
    // return rx_recv_comms_[dev];
  }
  void updateTxChannelActive(int channelId) { // hasn't been used yet
    int dev = channel2dev(channelId);
    int group_id = dev / NNIC_PER_GROUP;
    int offset = dev % NNIC_PER_GROUP;
    monitor_client_->markNicTxAsActive(group_id, offset);
  }
  void updateRxChannelActive(int channelId) { // hasn't been used yet
    int dev = channel2dev(channelId);
    int group_id = dev / NNIC_PER_GROUP;
    int offset = dev % NNIC_PER_GROUP;
    monitor_client_->markNicRxAsActive(group_id, offset);
  }
  uint32_t getTxUsage() {
    return 0xffffffff; // all available, place holder.
  }
  uint32_t refreshTxUsage() {
    return 0xffffffff; // all available, place holder.
  }
  // UNET: NIC selection related functions (monitor or manual) - end
  
  void* getMirrorSendComm(int channelId, int nNics) {
    int dev = channelId % nNics;
    return mirror_tx_send_comms_[dev];
  }
  void* getMirrorRxComm(int channelId, int nNics) {
    int dev = channelId % nNics;
    return mirror_rx_recv_comms_[dev];
  }

  UnetConnSetupState tx_setup_state_;
  UnetConnSetupState rx_setup_state_;
  std::vector<void*> tx_send_comms_; // side tx comms
  std::vector<void*> rx_recv_comms_; // side rx comms
  std::vector<void*> mirror_tx_send_comms_; // mirror tx comms
  std::vector<void*> mirror_rx_recv_comms_; // mirror rx comms
  std::vector<int> mirror_recv_wr_outstanding_; // outstanding recv WR count for each mirror comm
  struct sockaddr_in switch_addr_;// switch address
  int switch_sock_fd_;// switch socket file descriptor
private:

  bool MarkNicTxBusy(int dev, int channelId) {
    /*
      mark the nic busy after allocating the NIC to the channel.
      the allocation should be exclusive.
      This function should be called on two conditions:
      1. the primary NIC is occupied by the main channel
      2. occupy the NIC for side channel
      todo: this function should be called internally by the fuselink, not by other objects, private function
    */
    /*
    occupy through the interface of monitor
    */
    int group_id = dev / NNIC_PER_GROUP;
    int offset = dev % NNIC_PER_GROUP;
    monitor_client_->markNicTxAsActive(group_id, offset);
    return true;
  }
  bool MarkNicRxBusy(int dev, int channelId) {
    /* same as the tx version */
    int group_id = dev / NNIC_PER_GROUP;
    int offset = dev % NNIC_PER_GROUP;
    monitor_client_->markNicRxAsActive(group_id, offset);
    return true;
  }
  int channel2dev(int channelId) {
    return txchannel2dev_[channelId];
  }

  int ndevs_;
  int priority_dev_;
  int nchannels_;
  MonitorClient *monitor_client_;
  std::map<int, int> txchannel2dev_; // dev to comm.
  std::map<int, int> rxchannel2dev_; // dev to comm.
};

void UnetMemRegionInit(int nGPUs, void* base_addr, size_t size, int src_dev, int buffer_dev, struct UnetMemRegion *unmr, bool is_cuda_mem);

void UnetMemRegionDestroy(struct UnetMemRegion *unmr);

#endif

