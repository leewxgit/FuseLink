**流程**
监控网卡负载的模块
- 设置对应8个网卡的8个发送/接收队列
- 按照时间slot更新网卡的发送/接收状态
- 为了无锁操作，每个channel有独立的状态
- 每个time slot之后，根据channel的状态决定网卡的状态
- channel的状态是最后一次active的时间，timestamp
- 每当超出一个timeslot之后更新一下网卡的TX/RX状态

发送数据的模块

**建立连接**

数据类型
- 基本的连接，对应原来的connection实现，sendcomm, recvcomm里维护qp，以及fifo结构，这个作为default connection
- FuseLink连接，和原来的connection相比，不需要维护fifo
- FuseLink的连接manager

- 确认channel号
- 确认对应的group id, nic offset, channel offset
- 建立主连接
- 0号Channel建立FuseLink的连接
- Q:如何复用以前的结构？
- A:单纯的复用以前的nqp机制是不行的，因为以前的机制无法跨多个device运行，要想在多个设备上运行，需要能够抽象出一个在某个device上建立连接的过程

```c
fuselink_status {
  init,
  send,
  recv,
  finish
}

on_init()
on_send_qp()
on_recv_qp()
on_finish()

status connect(dev, socket, some_data, status)
```

现在的连接过程：
- 建立socket连接
- 准备verbs,也就是打开某个设备
- 注册fifo
- 建立qp
- 准备必要的qp info
- 发送qp info
- 接收qpinfo
- 调整qp状态

建立连接的核心结构：Verbs和qp

传输数据的核心结构：qp, cq, mr


FuseLink连接的建立过程
- 打开shm部分的状态
- 常规建立qp 连接
- 等待channel的同步
- channel同步完成之后开始使用网卡状态

**Register Memory**
- 检测当前设备处于哪个GPU
- 将原来的buffer 映射到对应的设备buffer上，原来的buffer先释放掉，记下这个buffer对应的GPU号

**irecv**
- 周期性地看是否有网卡空闲，如果有空闲网卡则占上，占上的过程需要加锁
- 如果网卡号和当前buffer所在的GPU号不匹配，要做重新映射
- 根据发送端的空闲情况决定是使用本地网卡还是使用相邻网卡作为目标网卡
- 针对目标网卡，post recv fifo, include网卡号
- 等待接收完成

**isend**
- 等待fifo信息
- 拿到fifo信息，解析网卡号，检测buffer和网卡号是否匹配，如果匹配就正常发，如果不匹配就记下来需要重新映射
- 发送，等待完成, imm data中include空闲网卡编号
- 完成后，如果网卡号和GPU号不匹配，要进行重新映射

**主channel进行通信**
- 和监控网卡负载的模块进行通信，告诉该模块需要占用这个网卡进行发送/接收
- 对于发送方来说，需要在post send的时候就告诉监控模块
- 对于接收方来说，需要在完成发送之后告诉监控模块

**测试**
[ ] 编译环境搭建
[ ] 建立连接过程
[ ] remapping过程