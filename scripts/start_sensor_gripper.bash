
#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
camera_fps=30
camera_width=640
camera_height=480
sensor_depth_camera_no=315122272335
gripper_depth_camera_no=315122271381

SENSOR_PATH_ID="usb-0:1.1:1.0-video-index0"
GRIPPER_PATH_ID="usb-0:2.1:1.0-video-index0"

# 自动获取 video 编号的函数
get_video_num() {
    local path_id=$1
    # 1. 在 /dev/v4l/by-path 中找包含 path_id 的行
    # 2. 提取最后面的 videoX
    # 3. 只保留数字部分
    local dev_path=$(ls -l /dev/v4l/by-path/ | grep "$path_id" | awk '{print $NF}')
    if [ -z "$dev_path" ]; then
        echo ""
    else
        echo ${dev_path//[^0-9]/}
    fi
}

# 动态获取端口号
sensor_fisheye_port=$(get_video_num "$SENSOR_PATH_ID")
gripper_fisheye_port=$(get_video_num "$GRIPPER_PATH_ID")

# 安全检查：如果没搜到则报错退出，防止 ROS 启动失败
if [ -z "$sensor_fisheye_port" ] || [ -z "$gripper_fisheye_port" ]; then
    echo "Error: 无法找到指定的摄像头物理端口！"
    exit 1
fi

echo "Detected Sensor Port: $sensor_fisheye_port"
echo "Detected Gripper Port: $gripper_fisheye_port"

sensor_serial_port=/dev/ttyUSB1
gripper_serial_port=/dev/ttyUSB0

export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.10/dist-packages:$PYTHONPATH

source /opt/ros/humble/setup.bash && cd $SCRIPT_DIR/../install/sensor_tools/share/sensor_tools/scripts/ && chmod 777 usb_camera.py
source $SCRIPT_DIR/../install/setup.bash && ros2 launch sensor_tools open_sensor_gripper.launch.py sensor_depth_camera_no:=_$sensor_depth_camera_no gripper_depth_camera_no:=_$gripper_depth_camera_no sensor_serial_port:=$sensor_serial_port gripper_serial_port:=$gripper_serial_port sensor_fisheye_port:=$sensor_fisheye_port gripper_fisheye_port:=$gripper_fisheye_port camera_fps:=$camera_fps camera_width:=$camera_width camera_height:=$camera_height camera_profile:=$camera_width,$camera_height,$camera_fps

