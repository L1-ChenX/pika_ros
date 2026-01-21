#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class GripperCmdControl(Node):
    def __init__(self):
        super().__init__('gripper_cmd_tester')
        # 注意：这里发布到 launch 文件暴露出的 /gripper_cmd
        self.pub = self.create_publisher(JointState, '/gripper_cmd', 10)
        self.get_logger().info('节点已启动: 发布到 /gripper_cmd')

    def move(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # 【关键】必须与 launch 文件中的 joint_name 一致
        msg.name = ['center_joint'] 
        
        # 目标位置 (通常 0.0=闭合, 0.08=张开)
        msg.position = [float(position)]
        
        # 填充速度和力度 (必须给值，否则可能无力)
        msg.velocity = [0.5]
        msg.effort = [1.0] 
        
        self.pub.publish(msg)
        self.get_logger().info(f'发送指令 -> {position}')

def main():
    rclpy.init()
    node = GripperCmdControl()
    try:
        print("等待连接...")
        time.sleep(1) # 等待发布者建立连接
        
        # 1. 尝试张开 (0.08m)
        print(">>> 尝试张开 (0.08)")
        node.move(0.08) 
        time.sleep(3)
        
        # 2. 尝试闭合 (0.0m)
        print(">>> 尝试闭合 (0.0)")
        node.move(0.0)
        time.sleep(3)
        
        # 3. 半开测试
        print(">>> 尝试半开 (0.04)")
        node.move(0.04)
        
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()