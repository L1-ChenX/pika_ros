"""
基于 ROS2 的 Pika 机器人实时推理控制脚本

使用说明：
1. 启动 Pika 硬件：
   cd ~/pika_ros/scripts/ && bash start_sensor_gripper.bash
   
2. 运行此脚本：
   python3 robot_control_ros2_pika.py
"""
import time
import numpy as np
import cv2
import argparse

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


class PikaRobotInterface(Node):
    """
    Pika 机器人 ROS2 硬件接口
    
    订阅的话题：
    - /gripper/camera/color/image_raw: RGB 摄像头
    - /puppet/end_pose_left: 末端位姿 (PoseStamped)
    - /joint_states_single: 尝试读取机械臂关节
    - /gripper/gripper/data: 夹爪状态
    
    发布的话题：
    - /pos_cmd: 机械臂控制 (PosCmd - 笛卡尔空间控制)
    - /gripper/gripper/ctrl: 夹爪控制
    
    服务：
    - /enable_srv: 使能机器人
    """
    
    def __init__(self):
        super().__init__('pika_robot_interface')
        
        # 数据缓存 (先初始化，防止后续报错导致属性不存在)
        self.latest_image = None
        self.latest_joint_positions = np.zeros(7, dtype=np.float64)
        self.latest_end_pose = np.zeros(6, dtype=np.float64)
        self.data_ready = False
        
        # 尝试导入自定义消息类型
        try:
            from piper_msgs.msg import PosCmd
            from piper_msgs.srv import Enable
            self.PosCmd = PosCmd
            self.Enable = Enable
            self.has_custom_msgs = True
        except ImportError:
            self.get_logger().error("无法导入 piper_msgs。请确保已 source ~/pika_ros/install/setup.bash")
            self.has_custom_msgs = False
            # 即使导入失败，也不要直接 return，让节点继续初始化以便输出错误日志
            # return
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # 订阅传感器数据
        self.get_logger().info("初始化 ROS2 订阅...")
        
        # RGB 摄像头订阅
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # 尝试订阅多个可能的关节状态话题
        # 优先订阅用户确认有效的 /joint_states_single
        self.create_subscription(JointState, '/joint_states_single', self.joint_callback, 10)
        # self.create_subscription(JointState, '/joint_states', self.joint_callback, 10) 
        # self.create_subscription(JointState, '/joint_states_single_gripper', self.joint_callback, 10)
        
        # 末端位姿订阅
        # self.pose_sub = self.create_subscription(
        #     PoseStamped,
        #     '/end_pose',
        #     self.pose_callback,
        #     10
        # )
        
        # 夹爪数据订阅
        self.gripper_sub = self.create_subscription(
            JointState,
            '/gripper/data',
            self.gripper_callback,
            10
        )
        
        # 发布控制指令 (机械臂)
        self.pos_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # 发布夹爪控制指令
        self.gripper_pub = self.create_publisher(
            JointState,
            '/gripper/ctrl',
            10
        )
        
        # 使能服务客户端
        self.enable_client = self.create_client(self.Enable, '/enable_srv')
        
        self.get_logger().info("✓ ROS2 接口初始化完成")
        
        # 尝试使能机器人
        self.enable_robot()
    
    def enable_robot(self):
        """调用服务使能机器人"""
        if not self.enable_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("未找到 /enable_srv 服务，跳过自动使能")
            return
            
        self.get_logger().info("正在请求使能机器人...")
        req = self.Enable.Request()
        req.enable_request = True
        
        future = self.enable_client.call_async(req)
        # 注意：在 spin 循环中不能直接 await，这里只是发送请求
        # 实际结果可以在回调中处理，或者简单地假设成功
        
    def image_callback(self, msg):
        """摄像头图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
    
    def joint_callback(self, msg):
        """关节状态回调 (处理多个可能的话题)"""
        if len(msg.position) >= 6:
            self.latest_joint_positions[:6] = np.array(msg.position[:6], dtype=np.float64)
            if len(msg.position) >= 7:
                 self.latest_joint_positions[6] = msg.position[6]
            self.data_ready = True

    def gripper_callback(self, msg):
        """夹爪状态回调"""
        if len(msg.position) > 0:
            self.latest_joint_positions[6] = msg.position[0]

    def pose_callback(self, msg):
        """末端位姿回调"""
        pose = msg.pose
        # 提取四元数
        q = pose.orientation
        
        # 简单的四元数转欧拉角 (XYZ 顺序)
        # roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2 # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        self.latest_end_pose = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            roll, pitch, yaw
        ], dtype=np.float64)
    
    def get_camera_image(self) -> np.ndarray:
        """获取摄像头图像"""
        if self.latest_image is None:
            self.get_logger().warn("尚未接收到摄像头图像")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.latest_image.copy()
    
    def get_state(self) -> np.ndarray:
        """获取机器人状态（13维）"""
        joint_positions = self.latest_joint_positions.copy()
        end_pose = self.latest_end_pose.copy()
        state = np.concatenate([joint_positions, end_pose])
        return state
    
    def execute_action(self, action: np.ndarray, motion_scale: float = 1.0):
        """
        执行动作 (Joint Space Control)
        Args:
            action: 13维向量 [7个关节, 6个末端位姿]
            motion_scale: 动作幅度缩放系数 (For joint control, usually 1.0)
        """
        # --- 1. 机械臂控制 (前6个关节) ---
        arm_msg = JointState()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        # 注意: 只有6个关节名称
        arm_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
        # 提取机械臂目标位置
        target_arm_joints = action[:6]
        arm_msg.position = target_arm_joints.tolist()
        arm_msg.velocity = []
        arm_msg.effort = []
        
        # self.pos_pub.publish(arm_msg)

        # --- 2. 夹爪控制 (第7个关节) ---
        gripper_msg = JointState()
        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        # 夹爪不需要特定名称，或者可以使用 'gripper'
        gripper_msg.name = ['gripper']
        
        # 提取夹爪目标位置
        target_gripper = action[6]
        gripper_msg.position = [target_gripper]
        gripper_msg.velocity = []
        gripper_msg.effort = []

        self.gripper_pub.publish(gripper_msg)
        
        # 调试日志
        if np.random.rand() < 0.1: # 降低频率
             self.get_logger().info(
                 f"发送控制 -> 机械臂(6): {target_arm_joints[:3]}..., 夹爪: {target_gripper:.3f}"
             )

    def is_ready(self) -> bool:
        """检查是否已接收到所有必要的传感器数据"""
        # 暂时放宽条件，只要有图像就认为就绪，方便调试
        if self.latest_image is not None:
             if not self.data_ready:
                 self.get_logger().warn("警告: 尚未接收到关节数据，但图像已就绪。尝试继续...")
             return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Pika Robot Control Client")
    parser.add_argument("--host", type=str, default="10.90.1.212", help="Inference server IP")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")
    parser.add_argument("--prompt", type=str, default="pick up the object", help="Task instruction for the robot")
    parser.add_argument("--scale", type=float, default=1.0, help="Motion scale factor (default: 2.0)")
    
    args = parser.parse_args()

    # ========== 配置参数 ==========
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    CONTROL_FREQUENCY = 10  # Hz
    PROMPT = args.prompt
    MOTION_SCALE = args.scale
    
    # ========== 初始化 ROS2 ==========
    print("=" * 60)
    print("Pika 机器人实时推理控制系统")
    print(f"Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"Task:   {PROMPT}")
    print(f"Scale:  {MOTION_SCALE}x")
    print("=" * 60)
    
    rclpy.init()
    robot = PikaRobotInterface()
    
    # ========== 控制循环 ==========
    print(f"\n开始控制循环 (频率: {CONTROL_FREQUENCY} Hz)")
    print("按 Ctrl+C 停止\n")
    
    loop_count = 0
    dt = 1.0 / CONTROL_FREQUENCY

    try:
        while rclpy.ok():
            loop_start = time.time()
            
            # 1. 处理 ROS2 回调
            rclpy.spin_once(robot, timeout_sec=0.001)
            
            action = np.array([0.2, 0.2, -0.2, 0.2, -0.2, 0.5, 0.02], dtype=np.float64)
            
            # 6. 执行动作（只执行第一个动作）
            try:
                robot.execute_action(action, motion_scale=MOTION_SCALE)
            except Exception as e:
                robot.get_logger().error(f"动作执行失败: {e}")
            time.sleep(0.02)
            
    
    except KeyboardInterrupt:
        print("\n\n收到停止信号，正在安全退出...")
    
    finally:
        # 清理资源
        robot.destroy_node()
        rclpy.shutdown()
        print("✓ 控制循环已停止")


if __name__ == "__main__":
    main()
