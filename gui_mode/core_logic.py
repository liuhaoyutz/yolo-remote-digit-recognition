# 核心逻辑分离文件 (core_logic.py)
"""
将原程序的核心逻辑重构为可复用的类
"""

import cv2
import threading
import queue
import time
import pandas as pd
from datetime import datetime
import os
import logging
from ultralytics import YOLO

class StreamReceiver(threading.Thread):
    """RTSP流接收线程"""
    def __init__(self, camera_url, frame_queue, logger=None):
        super().__init__()
        self.camera_url = camera_url
        self.q = frame_queue
        self.running = True
        self.cap = None
        self.logger = logger or logging.getLogger(__name__)
        
    def run(self):
        self.logger.info(f"开始接收视频帧: {self.camera_url}")
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.logger.info(f"正在连接 RTSP 流: {self.camera_url}")
                self.cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)
                time.sleep(1)

            ret, frame = self.cap.read()
            if ret:
                try:
                    if self.q.full():
                        self.q.get_nowait()  # 丢弃旧帧
                    self.q.put(frame.copy(), timeout=1.0)
                except (queue.Full, queue.Empty):
                    pass
            else:
                if self.running:
                    self.logger.warning("读取帧失败，正在重连...")
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)

            if not self.running:
                break

        if self.cap:
            self.cap.release()
            
class YOLOProcessor:
    """YOLO处理核心类"""
    def __init__(self, model_path, interval=2, min_val=None, max_val=None, 
                 display=True, excel_filename="data/detected_numbers.xlsx", logger=None):
        self.model_path = model_path
        self.interval = interval
        self.min_val = min_val
        self.max_val = max_val
        self.display = display
        self.excel_filename = excel_filename
        self.logger = logger or logging.getLogger(__name__)
        
        # 状态变量
        self.running = True
        self.q = queue.Queue(maxsize=2)
        self.model = None
        self.frame_count = 0
        self.detection_count = 0
        self.record_count = 0
        self.last_detection_time = 0
		
        # 记录状态
        self.last_recorded_number = None
        self.last_recorded_time = time.time()
        self.min_time_between_records = 60
        
        # 数据缓存
        self.data = []
        self.last_write_time = time.time()
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLO模型已加载: {self.model_path}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
            
    def process_frame(self, frame):
        """处理单个帧"""
        if not self.running:
            return None, None

        current_time = time.time()
        detected_value_for_excel = None
        displayed_number_str = ""

        # 是否执行检测
        should_detect = (self.interval == 0 or 
                        (current_time - self.last_detection_time >= self.interval))

        # 🔁 根据 display 选择是否处理图像
        processed_frame = None  # 默认不返回图像

        if should_detect:
            self.last_detection_time = current_time
            self.detection_count += 1

            results = self.model(frame)  # 直接在原始帧上推理
            digits = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls.item())
                    label = self.model.names[cls]
                    digits.append((x1, label, (x1, y1, x2, y2)))

            if digits:
                digits.sort(key=lambda d: d[0])
                digits = digits[:3]
                displayed_number_str = ''.join([d[1] for d in digits])
                try:
                    detected_value_for_excel = int(displayed_number_str)
                except ValueError:
                    detected_value_for_excel = None

            # ✅ 只有 display=True 时才绘制图像
            if self.display:
                processed_frame = frame.copy()  # 仅在需要显示时复制
                for x, label, bbox in digits:
                    cv2.rectangle(processed_frame, (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(processed_frame, label, (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # 转为 RGB 用于 PyQt 显示
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # 检查是否需要记录
        should_record = False
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if detected_value_for_excel is not None:
            # 值范围过滤
            if (self.min_val is not None and detected_value_for_excel < self.min_val) or \
               (self.max_val is not None and detected_value_for_excel > self.max_val):
                self.logger.info(f"数值 {detected_value_for_excel} 超出范围，已过滤")
                return processed_frame, None
                
            # 变化检测
            if (self.last_recorded_number is None or 
                str(detected_value_for_excel) != self.last_recorded_number or
                current_time - self.last_recorded_time >= self.min_time_between_records):
                should_record = True
                
        if should_record and detected_value_for_excel is not None:
            result = {
                "Frame": self.frame_count,
                "Time": current_time_str,
                "Detected Number": detected_value_for_excel
            }
            
            self.data.append(result)
            self.logger.info(f"记录检测结果: {detected_value_for_excel}")
            
            self.last_recorded_number = str(detected_value_for_excel)
            self.last_recorded_time = current_time
            self.record_count += 1
            
            # 定期写入Excel
            if current_time - self.last_write_time >= 60:
                self.write_to_excel()
                self.last_write_time = current_time
        else:
            result = None
            
        self.frame_count += 1
        
        return processed_frame, result
        
    def write_to_excel(self, data=None):
        """写入Excel文件"""
        data_to_write = data or self.data
        if not data_to_write:
            return
            
        try:
            os.makedirs(os.path.dirname(self.excel_filename), exist_ok=True)
            
            df = pd.DataFrame(data_to_write)
            
            if os.path.exists(self.excel_filename):
                with pd.ExcelWriter(self.excel_filename, mode='a', 
                                  engine='openpyxl', if_sheet_exists='overlay') as writer:
                    sheet_name = 'Sheet'
                    if sheet_name in writer.sheets:
                        start_row = writer.sheets[sheet_name].max_row
                        df.to_excel(writer, index=False, sheet_name=sheet_name, 
                                  startrow=start_row, header=False)
                    else:
                        df.to_excel(writer, index=False, sheet_name=sheet_name, 
                                  startrow=0, header=True)
            else:
                with pd.ExcelWriter(self.excel_filename, mode='w', 
                                  engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet', 
                              startrow=0, header=True)
                self.logger.info(f"已创建新的 Excel 文件: {self.excel_filename}")
                
            self.logger.info(f"已追加写入 {len(data_to_write)} 条数据")
            if data is None:
                self.data.clear()
                
        except Exception as e:
            self.logger.error(f"写入Excel失败: {e}")
            
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'data') and self.data:
            self.write_to_excel()