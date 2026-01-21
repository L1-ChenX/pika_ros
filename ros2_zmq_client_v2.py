import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import zmq
import pickle
import time
import threading  # <--- 引入多线程
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ================= 配置区域 =================
SERVER_IP = "10.90.1.212"
SERVER_PORT = 5555
CAMERA_MAPPING = {
    "main": "pikaGripperDepthCamera",
    "fisheye": "pikaGripperFisheyeCamera",
    "third": "pikaThirdPersonCamera"
}
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
# ===========================================

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class PiperRealClient(Node):
    def __init__(self):
        super().__init__('piper_real_client')
        self.bridge = CvBridge()
        
        # --- 数据缓存 (加锁保护) ---
        self.lock = threading.Lock()
        self.latest_joint = None
        self.latest_pose = None
        self.img_main = None
        self.img_fisheye = None
        self.img_third = None 
        
        # 目标位置 (从服务器获取)
        self.target_joints = None
        # 当前命令位置 (用于平滑插值)
        self.current_cmd_joints = None
        
        # --- QoS 设置 ---
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 订阅 ---
        self.create_subscription(JointState, '/joint_states_single_gripper', self.joint_cb, 10)
        self.create_subscription(Pose, '/end_pose', self.pose_cb, 10)
        self.create_subscription(Image, '/gripper/camera/color/image_raw', self.main_img_cb, video_qos)
        self.create_subscription(Image, '/gripper/camera_fisheye/color/image_raw', self.fisheye_img_cb, video_qos)
        self.create_subscription(Image, '/third_person/camera/color/image_raw', self.third_img_cb, video_qos)

        # --- 发布 ---
        # 请确保话题正确，如果是 sensor_tools 可能是 /joint_ctrl_single_gripper
        self.arm_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.gripper_pub = self.create_publisher(JointState, '/gripper_cmd', 10)

        # --- 启动网络线程 ---
        self.running = True
        self.net_thread = threading.Thread(target=self.network_loop)
        self.net_thread.daemon = True
        self.net_thread.start()

        # --- 启动控制循环 (高频: 50Hz) ---
        # 即使网络卡顿，这里也会以 50Hz 持续发送平滑指令，消除抖动
        self.create_timer(1.0/50.0, self.control_loop)
        
        self.get_logger().info("ROS 2 平滑控制节点已启动 (Threaded Mode)...")

    # ================= 回调函数 =================
    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            with self.lock:
                self.latest_joint = list(msg.position[:7])
                # 初始化当前命令位置为当前真实位置，防止启动时飞车
                if self.current_cmd_joints is None:
                    self.current_cmd_joints = list(msg.position[:7])

    def pose_cb(self, msg):
        p = msg.position
        o = msg.orientation
        r, p_ang, y = euler_from_quaternion(o.x, o.y, o.z, o.w)
        with self.lock:
            self.latest_pose = [p.x, p.y, p.z, r, p_ang, y]

    def main_img_cb(self, msg):
        try: 
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock: self.img_main = cv_img
        except Exception: pass

    def fisheye_img_cb(self, msg):
        try: 
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock: self.img_fisheye = cv_img
        except Exception: pass

    def third_img_cb(self, msg):
        try: 
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.lock: self.img_third = cv_img
        except Exception: pass

    # ================= 线程1: 网络通信 (慢) =================
    def network_loop(self):
        # 建立 ZMQ 连接 (只在线程内使用)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
        self.get_logger().info("ZMQ 线程已连接")

        while self.running and rclpy.ok():
            # 1. 取数据快照
            payload_data = None
            with self.lock:
                if any(d is None for d in [self.latest_joint, self.latest_pose, self.img_main, self.img_fisheye, self.img_third]):
                    time.sleep(0.1)
                    continue
                
                # 复制数据防止冲突
                my_joint = self.latest_joint[:]
                my_pose = self.latest_pose[:]
                my_img1 = self.img_main.copy()
                my_img2 = self.img_fisheye.copy()
                my_img3 = self.img_third.copy()

            # 2. 编码与发送
            try:
                obs_13d = my_joint + my_pose
                _, enc1 = cv2.imencode('.jpg', my_img1)
                _, enc2 = cv2.imencode('.jpg', my_img2)
                _, enc3 = cv2.imencode('.jpg', my_img3)

                payload = {
                    "images": {
                        CAMERA_MAPPING["main"]: enc1.tobytes(),
                        CAMERA_MAPPING["fisheye"]: enc2.tobytes(),
                        CAMERA_MAPPING["third"]: enc3.tobytes(),
                    },
                    "joint_state": obs_13d,
                    "gripper_state": [my_joint[-1]],
                    "text": "Grab the carrot and put it into the box."
                }
                
                start_t = time.time()
                socket.send(pickle.dumps(payload))
                
                # 这里会阻塞，但不会影响 control_loop
                resp = socket.recv()
                action = pickle.loads(resp)
                
                latency = (time.time() - start_t) * 1000
                # print(f"Network Latency: {latency:.1f}ms")

                if action is not None:
                    with self.lock:
                        # 更新最新的目标点
                        self.target_joints = action[:7] 

            except Exception as e:
                print(f"Network Error: {e}")
                time.sleep(1.0)

    # ================= 线程2: 控制循环 (快 - 50Hz) =================
    def control_loop(self):
        target = None
        with self.lock:
            if self.target_joints is None or self.current_cmd_joints is None:
                return
            target = self.target_joints[:]
        
        # --- 核心算法：平滑插值 (Exponential Smoothing) ---
        # 即使 target 不变，这个循环也会一直跑，保持机械臂稳定
        # alpha 越小越平滑，越大反应越快。0.1~0.2 适合消除抖动
        alpha = 0.15 
        
        new_cmd = []
        for i in range(7):
            # 公式: 当前 = 旧值 + alpha * (目标 - 旧值)
            val = self.current_cmd_joints[i] + alpha * (target[i] - self.current_cmd_joints[i])
            new_cmd.append(val)
        
        # 更新当前命令值
        self.current_cmd_joints = new_cmd

        # --- 发布消息 ---
        # 1. 机械臂 (1-6轴)
        arm_msg = JointState()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.header.frame_id = "piper_single"
        arm_msg.name = JOINT_NAMES
        arm_msg.position = [float(x) for x in new_cmd]
        arm_msg.velocity = [] # 留空，不要锁死速度
        arm_msg.effort = []
        self.arm_pub.publish(arm_msg)

        # 2. 夹爪 (第7轴)
        gripper_msg = JointState()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.name = ['center_joint']
        gripper_msg.position = [float(new_cmd[6])]
        gripper_msg.velocity = [0.5]
        gripper_msg.effort = [1.0]
        self.gripper_pub.publish(gripper_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PiperRealClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()