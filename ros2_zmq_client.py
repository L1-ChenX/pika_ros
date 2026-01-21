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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ================= 配置区域 =================
SERVER_IP = "10.90.1.212"
SERVER_PORT = 5555

# 必须与 Server 端的训练配置一致
CAMERA_MAPPING = {
    "main": "pikaGripperDepthCamera",       # 主摄
    "fisheye": "pikaGripperFisheyeCamera",  # 鱼眼
    "third": "pikaThirdPersonCamera"        # 第三人称
}

# 机械臂关节名称 (joint1-6)
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
# ===========================================

def euler_from_quaternion(x, y, z, w):
    """四元数转欧拉角"""
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
        
        # --- ZMQ 连接 ---
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.get_logger().info(f"正在连接到服务器 {SERVER_IP}:{SERVER_PORT} ...")
        self.socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
        
        # --- 数据缓存 ---
        self.latest_joint = None
        self.latest_pose = None
        self.img_main = None
        self.img_fisheye = None
        self.img_third = None 
        
        # --- QoS 设置 ---
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 订阅 (传感器输入) ---
        self.create_subscription(JointState, '/joint_states_single_gripper', self.joint_cb, 10)
        self.create_subscription(Pose, '/end_pose', self.pose_cb, 10)
        
        # 相机订阅
        self.create_subscription(Image, '/gripper/camera/color/image_raw', self.main_img_cb, video_qos)
        self.create_subscription(Image, '/gripper/camera_fisheye/color/image_raw', self.fisheye_img_cb, video_qos)
        self.create_subscription(Image, '/third_person/camera/color/image_raw', self.third_img_cb, video_qos)

        # --- 发布 (机器人控制) ---
        
        # 1. 机械臂控制发布者 (保持原样)
        self.arm_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 2. [新增] 夹爪控制发布者 -> /gripper_cmd
        self.gripper_pub = self.create_publisher(JointState, '/gripper_cmd', 10)

        # --- 循环 ---
        self.create_timer(1.0/30.0, self.inference_loop)
        self.get_logger().info("ROS 2 控制节点启动 (Arm + GripperCmd Mode)...")

    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            self.latest_joint = list(msg.position[:7])

    def pose_cb(self, msg):
        p = msg.position            # Pose 直接访问 .position
        o = msg.orientation         # Pose 直接访问 .orientation
        r, p_ang, y = euler_from_quaternion(o.x, o.y, o.z, o.w)
        self.latest_pose = [p.x, p.y, p.z, r, p_ang, y]

    def main_img_cb(self, msg):
        try: self.img_main = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e: self.get_logger().error(f"Main Img Error: {e}")

    def fisheye_img_cb(self, msg):
        try: self.img_fisheye = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e: self.get_logger().error(f"Fisheye Img Error: {e}")

    def third_img_cb(self, msg):
        try: self.img_third = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e: self.get_logger().error(f"Third Img Error: {e}")

    def inference_loop(self):
        # 1. 检查数据齐备性
        if any(d is None for d in [self.latest_joint, self.latest_pose, self.img_main, self.img_fisheye, self.img_third]):
            return

        try:
            start_time = time.time()

            # 2. 构造数据包
            obs_13d = self.latest_joint + self.latest_pose
            
            _, enc_main = cv2.imencode('.jpg', self.img_main)
            _, enc_fisheye = cv2.imencode('.jpg', self.img_fisheye)
            _, enc_third = cv2.imencode('.jpg', self.img_third)

            payload = {
                "images": {
                    CAMERA_MAPPING["main"]: enc_main.tobytes(),
                    CAMERA_MAPPING["fisheye"]: enc_fisheye.tobytes(),
                    CAMERA_MAPPING["third"]: enc_third.tobytes(),
                },
                "joint_state": obs_13d,
                "gripper_state": [self.latest_joint[-1]],
                "text": "Grab the carrot and put it into the box."
            }

            # 3. 发送与接收
            self.socket.send(pickle.dumps(payload))
            response_bytes = self.socket.recv()
            action = pickle.loads(response_bytes)
            
            latency = (time.time() - start_time) * 1000

            # 4. 执行控制
            if action is not None:
                # action: [joint1, ..., joint6, gripper]
                target_positions = action[:7] 

                print(f"[{latency:.1f}ms] Target: {np.round(target_positions, 3)}")

                # ================= 机械臂控制 (Joint 1-6) =================
                # 注意：虽然这里发了 joint7，但因为有专门的 gripper_cmd，
                # 机械臂控制器可能会忽略第7轴，或者两者共存。
                arm_msg = JointState()
                arm_msg.header.stamp = self.get_clock().now().to_msg()
                arm_msg.header.frame_id = "piper_single"
                arm_msg.name = JOINT_NAMES # ['joint1'...'joint7']
                
                # 填充位置
                arm_msg.position = [float(x) for x in target_positions]
                # 填充速度和力矩 (防止机械臂无力)
                arm_msg.velocity = [] 
                arm_msg.effort = []
                
                # 发布机械臂消息
                self.arm_pub.publish(arm_msg)

                # ================= [核心修改] 夹爪控制 (/gripper_cmd) =================
                gripper_val = float(target_positions[6]) # 获取第7维数据作为夹爪值
                
                gripper_msg = JointState()
                gripper_msg.header.stamp = self.get_clock().now().to_msg()
                
                # 【关键】名称必须是 center_joint
                gripper_msg.name = ['center_joint'] 
                
                # 设置目标位置
                gripper_msg.position = [gripper_val]
                
                # 【关键】设置速度和力度 (根据你验证成功的参数)
                gripper_msg.velocity = [0.5]
                gripper_msg.effort = [1.0]

                # 发布夹爪独立消息
                self.gripper_pub.publish(gripper_msg)
                # ====================================================================
                
            else:
                self.get_logger().warn("❌ Server returned None")

        except zmq.ZMQError as e:
            self.get_logger().error(f"ZMQ Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PiperRealClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()