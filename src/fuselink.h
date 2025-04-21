#ifndef FUSELINK_H
#define FUSELINK_H
// #include <infiniband/verbs.h>
#include "ibvwrap.h"
#include <vector>
#include <map>
#include "monitor.h"

#include "nccl_net.h" // interaction with nccl net plugin

extern ncclNet_t ncclNetPlugin_v7;

// enum FuseLinkConnState {
//   FuseLinkConnStateStart,
//   FuseLinkConnStateConnect,
//   FuseLinkConnStateAccept,
//   FuseLinkConnStateSend,
//   FuseLinkConnStateRecv,
//   FuseLinkConnStateConnecting,
//   FuseLinkConnStateConnected,
//   FuseLinkConnStatePendingReady
// };

enum FuseLinkConnSetupState {
  FuseLinkConnSetupStateInit = 0,
  FuseLinkConnSetupStatePending = 1,
  FuseLinkConnSetupStateReady = 2
};

// struct FuseLinkConnection {
//   int dev;
//   ibv_qp *qp;
//   ibv_cq *cq;
// };

class FuseLinkConnManager {
public:
  FuseLinkConnManager(int ndevs, int priority_dev) {
    ndevs_ = ndevs;
    priority_dev_ = priority_dev;
  }
  FuseLinkConnManager(int ndevs, int priority_dev, int nchannels) {
    ndevs_ = ndevs;
    priority_dev_ = priority_dev;
    nchannels_ = nchannels;
    for (int i = 0; i < nchannels; ++i) {
      txchannel2dev_[i] = priority_dev_;
      rxchannel2dev_[i] = priority_dev_;
    }
  }
  int GetNdevs() {
    return ndevs_;
  }
  ~FuseLinkConnManager() {
    // tear down all connections and monitor
    for (int i = 0; i < ndevs_; i++) {
      if (tx_send_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeSend(tx_send_comms_[i]);
      }
      if (rx_recv_comms_[i] != NULL) {
        ncclNetPlugin_v7.closeRecv(rx_recv_comms_[i]);
      }
    }
    // delete monitor
    delete monitor_client_;
  }
  void SetUpMonitor() {
    monitor_client_ = new MonitorClient();
  }
  // FuseLinkConnection* GetAssignedDev(int channelId) {
  //   // return with assigned dev
  // }

  // FuseLinkConnection* ConnTx(int i) {
  //   return &tx_connections_[i];
  // }

  // FuseLinkConnection* ConnRx(int i) {
  //   return &rx_connections_[i];
  // }

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

  bool MarkTxChannelActive(int channelId) {
    return MarkNicTxBusy(channel2dev(channelId), channelId);
  }

  bool MarkRxChannelActive(int channelId) {
    return MarkNicRxBusy(channel2dev(channelId), channelId);
  }

  void* RefreshTxComm(int channelId) {
    /*
      this function is called by the channel to get a NIC for tx
      each channel independently call this function, so this function should be implemented with thread safety
    */
    // if (channelId / 2 == 0) {
    //   // main channel
    //   return NULL;
    // } else {
    //   // side channel
    //   return tx_send_comms_[channel2dev(channelId)];
    // }
    return NULL;
  }

  void* bidTxChannel(int channelId, int txNicId) {
    // DEBUG: return NULL to enforce the use of comm itself
    return tx_send_comms_[txNicId];
    // DEBUG: end
    int group_id = txNicId / NNIC_PER_GROUP;
    int offset = txNicId % NNIC_PER_GROUP;
    return tx_send_comms_[txNicId];
    // return monitor_client_->bidTxChannel(group_id, offset);
  }

  bool RefreshRxComm(int channelId) {
    /* same as the tx version */
    if (channelId / 2 == 0) {
      return NULL;
    } else {
      return rx_recv_comms_[channel2dev(channelId)];
    }
  }

  int RegisterMem(void* ptr, size_t size, void** handle) {
    /*
      register the memory on all devices, we need an array of mr, use on demand
    */
    // not implemented yet
    return -1;
  }

  void* channel2txcomm(int channelId) {
    int dev = channel2dev(channelId);
    return tx_send_comms_[dev];
  }

  void* channel2rxcomm(int channelId) {
    int dev = channel2dev(channelId);
    return rx_recv_comms_[dev];
  }

  void* refreshTxComm(int channelId) {
    // call monitorclient to get a new nic
    int nic_id = txchannel2dev_[channelId];
    int group_id = nic_id / NNIC_PER_GROUP;
    int offset = nic_id % NNIC_PER_GROUP;
    // todo: update idle nic tx arguments.
    int dev = monitor_client_->getIdleNicTx(group_id, priority_dev_);
    txchannel2dev_[channelId] = dev;
    return tx_send_comms_[dev];
  }

  void* refreshRxComm(int channelId) {
    // call monitorclient to get a new nic
    int group_id = channelId / NNIC_PER_GROUP;
    int offset = channelId % NNIC_PER_GROUP;
    int dev = monitor_client_->getIdleNicRx(group_id, priority_dev_, 0xffffffff);
    rxchannel2dev_[channelId] = dev;
    return rx_recv_comms_[dev];
  }

  void* refreshRxComm(int channelId, int* txAvailable, int nNics) {
    // call monitorclient to get a new nic
    // convert txAvailable to mask

    // DEBUG: return NULL to enforce the use of the comm itself
    return NULL;
    // DEBUG: end

    int mask = 0;
    for (int i = 0; i < nNics; ++i) {
      if (txAvailable[i]) {
        mask |= (1 << i);
      }
    }
    int group_id = rxchannel2dev_[channelId] / NNIC_PER_GROUP;
    int dev = monitor_client_->getIdleNicRx(group_id, priority_dev_, mask);
    rxchannel2dev_[channelId] = dev;
    return rx_recv_comms_[dev];
  }

  void* refreshRxComm(int channelId, uint32_t txAvailable, int nNics, int* fuselink_offset) {
    // call monitorclient to get a new nic
    // convert txAvailable to mask

    // DEBUG: return NULL to enforce the use of the comm itself
    // randomly return a comm
    int dev = 1;
    printf("refreshRxComm: %d len: %d\n", dev, rx_recv_comms_.size());
    *fuselink_offset = dev;
    return rx_recv_comms_[dev];
    // DEBUG: end

    int group_id = channelId / NNIC_PER_GROUP;

    dev = monitor_client_->getIdleNicRx(group_id, priority_dev_, txAvailable);
    rxchannel2dev_[channelId] = dev;
    return rx_recv_comms_[dev];
  }


  void updateTxChannelActive(int channelId) { // called by the channel
    int dev = channel2dev(channelId);
    int group_id = dev / NNIC_PER_GROUP;
    int offset = dev % NNIC_PER_GROUP;
    monitor_client_->markNicTxAsActive(group_id, offset);
  }

  void updateRxChannelActive(int channelId) {
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

  FuseLinkConnSetupState tx_setup_state_;
  FuseLinkConnSetupState rx_setup_state_;
  std::vector<void*> tx_send_comms_; // actually tx channels
  std::vector<void*> rx_recv_comms_; // actually rx channels
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
  // std::vector<FuseLinkConnection> tx_connections_;
  // std::vector<FuseLinkConnection> rx_connections_;
  int nchannels_;

  MonitorClient *monitor_client_;
  std::map<int, int> txchannel2dev_; // dev to comm.
  std::map<int, int> rxchannel2dev_; // dev to comm.
};


#endif

