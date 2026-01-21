"""
client_mock.py (Client B - Simulator)
模拟向 Server A 发送图像和状态数据，并打印返回的动作。
"""
import zmq
import pickle
import numpy as np
import cv2
import time
import random

# ================= 配置区域 =================
SERVER_IP = "10.90.1.212"
PORT = 5555

# 必须与 Server A 的 CAMERA_MAPPING 键名完全一致
CAMERA_KEYS = [
    "pikaGripperDepthCamera",
    "pikaGripperFisheyeCamera",
    "pikaThirdPersonCamera",
]

# 模拟图像尺寸 (H, W, C) - 请根据你的实际相机调整，或者用默认的
IMG_SHAPE = (480, 640, 3) 
# ===========================================

def generate_dummy_image():
    """生成一张随机噪点的假图片，并编码为 bytes"""
    # 生成 0-255 的随机像素
    img = np.random.randint(0, 256, IMG_SHAPE, dtype=np.uint8)
    # Server 端使用 cv2.imdecode，所以这里必须编码成 jpg 或 png 格式的 bytes
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()

def main():
    context = zmq.Context()
    #以此处使用 REQ (Request) 模式，对应服务端的 REP (Reply)
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{SERVER_IP}:{PORT}")
    
    print(f"Client: Connected to server at {SERVER_IP}:{PORT}")
    print("Client: Starting simulation loop (Press Ctrl+C to stop)...")

    try:
        step = 0
        while True:
            start_time = time.time()

            # --- 1. 构造模拟 Payload ---
            payload = {
                "images": {},
                "joint_state": [],
                "gripper_state": [],
                "text": "Grab the carrot and put it into the box."
            }

            # 填充图像数据
            for key in CAMERA_KEYS:
                payload["images"][key] = generate_dummy_image()

            # 填充机械臂状态 (模拟 6个关节角度)
            # 这里的数值只是为了测试格式，你可以改成正弦波让它动起来
            payload["joint_state"] = np.random.uniform(-0.5, 0.5, size=(6,)).tolist()
            
            # 填充夹爪状态 (模拟 1个开合度)
            payload["gripper_state"] = [random.random()] # 0.0 ~ 1.0

            # --- 2. 序列化并发送 ---
            print(f"\n[Step {step}] Sending request...", end="", flush=True)
            socket.send(pickle.dumps(payload))

            # --- 3. 等待并接收响应 ---
            #这一步是阻塞的，必须等到 Server 返回才能继续
            response_bytes = socket.recv()
            action = pickle.loads(response_bytes)

            end_time = time.time()
            latency = (end_time - start_time) * 1000

            # --- 4. 打印结果 ---
            print(f" Done! ({latency:.1f}ms)")
            
            if action is None:
                print("❌ Server returned None (Error occurred on server)")
            else:
                # 格式化打印动作向量
                action_str = ", ".join([f"{x:.4f}" for x in action])
                print(f"✅ Received Action ({len(action)} dim): [{action_str}]")

            # 模拟控制频率 (例如 10Hz)
            time.sleep(0.1) 
            step += 1

    except KeyboardInterrupt:
        print("\nClient: Stopped by user.")
    except Exception as e:
        print(f"\nClient: Error - {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()