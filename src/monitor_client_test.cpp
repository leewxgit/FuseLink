#include "monitor.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

class MonitorClientTest : public ::testing::Test {
protected:
    Monitor* monitor;
    MonitorClient* client1;
    MonitorClient* client2;

    void SetUp() override {
        monitor = new Monitor();
        ASSERT_TRUE(monitor->initialize());
        monitor->start();
        
        // Give monitor some time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        client1 = new MonitorClient();
        client2 = new MonitorClient();
    }

    void TearDown() override {
        delete client1;
        delete client2;
        // monitor->stop();
        delete monitor;
    }
};

TEST_F(MonitorClientTest, TwoClientsCanAccessSharedMemory) {
    // Client 1 marks a NIC as active
    client1->markNicTxAsActive(0, 0);
    
    // Give some time for the status to be updated
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    
    // Client 2 should be able to see the same status
    NicStatus status[1];
    client2->getNICStatus(1, status);
    EXPECT_EQ(status[0], NIC_STATUS_BUSY);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // client1->markNicTxAsActive(0, 0);
    // std::this_thread::sleep_for(std::chrono::microseconds(10));
    client2->getNICStatus(1, status);
    EXPECT_EQ(status[0], NIC_STATUS_IDLE);
}

// TEST_F(MonitorClientTest, IdleNicDetection) {
//     // Initially all NICs should be idle
//     int idle_nic = client1->getIdleNicTx(0, 0);
//     EXPECT_NE(idle_nic, -1);
    
//     // Mark a NIC as active
//     client1->markNicTxAsActive(0, 0);
    
//     // Try to get an idle NIC again
//     int new_idle_nic = client2->getIdleNicTx(0, 0);
//     EXPECT_NE(new_idle_nic, idle_nic); // Should get a different NIC
// }

TEST_F(MonitorClientTest, ConcurrentAccess) {
    // Test concurrent access from both clients
    std::thread t1([this]() {
        for (int i = 0; i < 10; ++i) {
            client1->markNicTxAsActive(0, 0);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });
    
    std::thread t2([this]() {
        for (int i = 0; i < 10; ++i) {
            client2->markNicRxAsActive(0, 0);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });
    
    t1.join();
    t2.join();
    
    // Verify final status
    NicStatus status[1];
    client1->getNICStatus(1, status);
    EXPECT_EQ(status[0], NIC_STATUS_BUSY);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 