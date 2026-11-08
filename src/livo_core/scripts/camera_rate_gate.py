#!/usr/bin/env python3
"""Deliver up to 10 coherent camera batches/sec, preserving sensor stamps."""
from collections import deque
import json
import struct
from std_msgs.msg import String
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import Image

class Gate(Node):
    def __init__(self):
        super().__init__('qvs_camera_rate_gate')
        self.buffers={};self.arrivals={};self.stamps={};self.pubs={};self.subs=[]
        self.received={};self.published={}
        self.diag_pub=self.create_publisher(String,"/qvs/input/gate_diagnostics",2)
        self.diag_timer=self.create_timer(5.0,self.diagnostics)
        qos=QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)
        for name,ns in [('front','camera'),('left','left'),('right','right'),('back','back')]:
            self.buffers[name]=deque(maxlen=8)
            self.received[name]=0;self.published[name]=0
            self.pubs[name]=self.create_publisher(Image,f'/qvs/input/{name}/image',qos)
            self.subs.append(self.create_subscription(Image,f'/{ns}/camera/color/image_raw',lambda m,n=name:self.receive(n,m),qos,raw=True))
        self.timer=self.create_timer(.1,self.tick)
        self.get_logger().info('10 Hz coherent image gate; 35 ms batch span; original stamps preserved')
    def diagnostics(self):
        d={n:{'received':self.received[n],'published':self.published[n],
              'not_forwarded':self.received[n]-self.published[n]-len(self.buffers[n]),
              'buffered':len(self.buffers[n]),'last_source_ns':str(self.stamps.get(n,0))}
           for n in self.buffers}
        msg=String();msg.data=json.dumps({'target_hz':10,'source_stamps_preserved':True,'cameras':d})
        self.diag_pub.publish(msg)
        self.get_logger().info(msg.data)
    def receive(self,name,msg):
        self.received[name]+=1
        sec,nsec=struct.unpack_from('<iI' if msg[1]&1 else '>iI',msg,4)
        stamp=sec*1_000_000_000+nsec
        if stamp>self.stamps.get(name,-1):self.buffers[name].append((stamp,msg))
        self.arrivals[name]=time.monotonic()
    def tick(self):
        now=time.monotonic()
        # A missing camera must not block the other healthy cameras.
        names=[k for k in self.buffers if now-self.arrivals.get(k,-1e9)<.15]
        if not names:return
        buffers={k:[x for x in self.buffers[k] if x[0]>self.stamps.get(k,-1)] for k in names}
        if any(not v for v in buffers.values()):return
        ceilings=sorted({t for b in buffers.values() for t,_ in b},reverse=True)
        for ceiling in ceilings:
            selected={}
            for name,buf in buffers.items():
                matches=[x for x in buf if ceiling-35_000_000<=x[0]<=ceiling]
                if matches:selected[name]=max(matches,key=lambda x:x[0])
            if len(selected)!=len(names):continue
            for name,(stamp,msg) in selected.items():
                self.pubs[name].publish(msg);self.stamps[name]=stamp;self.published[name]+=1
                self.buffers[name]=deque((x for x in self.buffers[name] if x[0]>stamp),maxlen=8)
            return

rclpy.init();node=Gate()
try:rclpy.spin(node)
except KeyboardInterrupt:pass
finally:node.destroy_node();rclpy.try_shutdown()
