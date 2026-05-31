from ultralytics import YOLO
import cv2
import signal
import argparse
import time
import pandas as pd
from datetime import datetime
import os
import queue
import threading
import sys
import logging

# 全局变量
running = True
q = queue.Queue(maxsize=2)  # 防止内存堆积
logger = None  # 日志全局变量

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global running
    logger.warning("接收到 Ctrl+C，正在安全退出...")
    running = False
    try:
        q.put_nowait(None)
    except:
        pass

def setup_logging(excel_filename):
    """
    配置日志：日志文件与 Excel 同名，仅扩展名不同，保存在 log/ 目录下
    """
    # 从 Excel 文件名生成日志文件名
    base_name = os.path.basename(excel_filename)
    log_filename = base_name.replace(".xlsx", ".txt")
    log_path = os.path.join("log", log_filename)
    
    # 创建 log 目录
    os.makedirs("log", exist_ok=True)

    # 清除之前的 handlers，避免重复日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),  # 覆盖写入
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def write_to_excel(data, filename):
    """
    将数据追加写入 Excel 文件，确保第一行是表头
    """
    if not data:
        return

    df = pd.DataFrame(data)

    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if os.path.exists(filename):
            # 文件已存在：追加数据，不写 header
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                sheet_name = 'Sheet'
                if sheet_name in writer.sheets:
                    start_row = writer.sheets[sheet_name].max_row
                    df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_row, header=False)
                else:
                    # 安全兜底：如果 sheet 不存在，写 header
                    df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=0, header=True)
        else:
            # 文件不存在：创建并写 header
            with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet', startrow=0, header=True)
            logger.info(f"已创建新的 Excel 文件: {filename}")

        logger.info(f"已追加写入 {len(data)} 条数据到 Excel 文件")
    except Exception as e:
        logger.error(f"写入 Excel 时出错：{e}")

def receive():
    """从RTSP流接收视频帧（带超时设置和自动重连）"""
    global camera_url
    logger.info("开始接收视频帧")
    cap = None

    while running:
        if cap is None or not cap.isOpened():
            logger.info(f"正在连接 RTSP 流: {camera_url}")
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)
            time.sleep(1)

        ret, frame = cap.read()
        if ret:
            try:
                if q.full():
                    q.get_nowait()  # 丢弃旧帧
                q.put(frame.copy(), timeout=1.0)
            except (queue.Full, queue.Empty):
                pass  # 忽略队列满或空的情况
        else:
            if running:
                logger.warning("读取帧失败，可能是流中断或超时，正在重连...")
            if cap:
                cap.release()
            cap = None
            time.sleep(1)

        if not running:
            break

    if cap:
        cap.release()

def process(excel_filename):
    """处理并显示接收到的视频帧"""
    global running
    logger.info(f"图像显示: {'启用' if args.display else '禁用'}")

    # 加载 YOLO 模型
    model = YOLO('runs\\detect\\yolo11n_200_epochs_base_video_1234\\train\\weights\\best.pt')

    last_recorded_number = None
    last_recorded_time = time.time()
    min_time_between_records = 60  # 最小记录间隔（秒）
    data = []
    frame_count = 0
    last_write_time = time.time()
    detection_interval = args.interval
    last_detection_time = time.time()

    logger.info(f"Excel 文件将保存为：{excel_filename}")

    while running or not q.empty():
        try:
            frame = q.get(timeout=1.0)
            if frame is None:
                break
        except queue.Empty:
            if not running:
                break
            continue

        current_time = time.time()
        detected_value_for_excel = None
        displayed_number_str = ""

        # 是否执行检测
        if detection_interval == 0 or (current_time - last_detection_time >= detection_interval):
            last_detection_time = current_time

            results = model(frame)
            digits = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls.item())
                    label = model.names[cls]
                    digits.append((x1, label, (x1, y1, x2, y2)))

            if digits:
                digits.sort(key=lambda d: d[0])  # 按 x 坐标排序
                digits = digits[:3]  # 取前三位
                displayed_number_str = ''.join([d[1] for d in digits])
                try:
                    detected_value_for_excel = int(displayed_number_str)
                except ValueError:
                    detected_value_for_excel = None

                # 绘制检测框和标签
                for x, label, bbox in digits:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        should_record = False

        if detected_value_for_excel is not None:
            # 检查最小值限制
            if args.min is not None and detected_value_for_excel < args.min:
                logger.info(f"帧号: {frame_count} | 时间: {current_time_str} | 数字 {detected_value_for_excel} 小于最小值 {args.min}，忽略")
                continue
            # 检查最大值限制
            if args.max is not None and detected_value_for_excel > args.max:
                logger.info(f"帧号: {frame_count} | 时间: {current_time_str} | 数字 {detected_value_for_excel} 大于最大值 {args.max}，忽略")
                continue

            # 判断是否需要记录
            if last_recorded_number is None:
                should_record = True
            elif str(detected_value_for_excel) != last_recorded_number:
                should_record = True
            elif current_time - last_recorded_time >= min_time_between_records:
                should_record = True

        if should_record and detected_value_for_excel is not None:
            data.append({
                "Frame": frame_count,
                "Time": current_time_str,
                "Detected Number": detected_value_for_excel
            })
            logger.info(f"帧号: {frame_count} | 时间: {current_time_str}")
            logger.info(f"检测到的数字：{displayed_number_str}")
            last_recorded_number = str(detected_value_for_excel)
            last_recorded_time = current_time

        # 显示图像（可选）
        if args.display:
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('YOLO Real-Time Detection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not running:
                logger.info("正在退出...")
                running = False
                break

        frame_count += 1

        # 每分钟写入一次 Excel（防止频繁 I/O）
        if current_time - last_write_time >= 60:
            if data:
                write_to_excel(data, excel_filename)
                data.clear()
                last_write_time = current_time

    # 程序结束前写入剩余数据
    if data:
        write_to_excel(data, excel_filename)

    cv2.destroyAllWindows()

# ============ 主程序入口 ============
if __name__ == '__main__':
    # 生成统一的时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"detected_numbers_{timestamp}"

    # 定义文件路径
    data_dir = "data"
    excel_filename = os.path.join(data_dir, f"{base_filename}.xlsx")

    # 初始化日志（必须在最前面）
    logger = setup_logging(excel_filename)

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO 实时检测网络摄像头（仅识别数字）")
    parser.add_argument("--ip", type=str, help="网络摄像头的IP地址", default="192.168.31.1")
    parser.add_argument("--interval", type=int, default=2, help="检测间隔时间（秒），0 表示每帧都检测")
    parser.add_argument("--min", type=int, help="只记录大于等于该值的数字，不设置则无下限")
    parser.add_argument("--max", type=int, help="只记录小于等于该值的数字，不设置则无上限")
    parser.add_argument("--display", action='store_true', help="显示检测图像窗口（默认开启）")
    parser.add_argument("--no-display", dest='display', action='store_false', help="不显示检测图像窗口")
    parser.set_defaults(display=True)
    args = parser.parse_args()

    # 构建 RTSP URL
    ip_address = args.ip
    camera_url = f"rtsp://{ip_address}:8554/stream0"
    logger.info(f"RTSP 流地址: {camera_url}")

    # 启动线程
    t_receive = threading.Thread(target=receive, name="RTSP_Receiver")
    t_process = threading.Thread(target=process, args=(excel_filename,), name="YOLO_Processor")

    t_receive.start()
    t_process.start()

    try:
        logger.info("按 Ctrl+C 或在图像窗口按 'q' 键退出程序...")
        while t_receive.is_alive() and t_process.is_alive():
            if not running:
                logger.info("检测到退出信号，正在等待线程结束...")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.warning("检测到 KeyboardInterrupt，正在退出...")
        running = False
    except Exception as e:
        logger.error(f"主线程异常: {e}")
    finally:
        # 安全退出线程
        t_receive.join(timeout=3)
        t_process.join(timeout=3)
        logger.info("程序已安全退出。")