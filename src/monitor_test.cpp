#include "monitor.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

class MonitorTest : public ::testing::Test {
protected:
    Monitor* monitor;

    void SetUp() override {
        monitor = new Monitor();
        ASSERT_TRUE(monitor->initialize());
    }

    void TearDown() override {
        delete monitor;
    }
};

TEST_F(MonitorTest, BasicInitialization) {
    monitor->start();
    EXPECT_TRUE(monitor->getNicCount() > 0);
}

TEST_F(MonitorTest, StartStop) {
    monitor->start();
    std::cout << "Monitor started" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // monitor->stop();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 