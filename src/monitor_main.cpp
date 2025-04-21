#include "monitor.h"
#include <chrono>
#include <thread>

int main() {
  Monitor *monitor = new Monitor();
  monitor->start();
  std::this_thread::sleep_for(std::chrono::seconds(10));
  monitor->stop();
  delete monitor;

  return 0;
}
