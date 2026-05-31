# gui_main.py
"""
YOLO 数字识别系统 - 图形用户界面
基于 PyQt5 开发
"""

import sys
import os
import cv2
import queue
import traceback
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QSpinBox, QFileDialog,
    QGroupBox, QTextEdit, QProgressBar, QTabWidget, QMessageBox,
    QFormLayout, QComboBox, QFrame, QSizePolicy, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import logging

# 导入核心程序逻辑（稍作修改）
from core_logic import YOLOProcessor, StreamReceiver

# 自定义信号发射器
class LogEmitter(QObject):
    message = pyqtSignal(str)

# 自定义 Handler
class QtHandler(logging.Handler):
    def __init__(self, log_emitter):
        super().__init__()
        self.log_emitter = log_emitter

    def emit(self, record):
        msg = self.format(record)
        self.log_emitter.message.emit(msg)

class YOLODetectionThread(QThread):
    """处理YOLO检测的线程"""
    frame_ready = pyqtSignal(object)  # 发送处理后的帧
    detection_result = pyqtSignal(dict)  # 发送检测结果
    status_update = pyqtSignal(str)  # 发送状态信息
    # progress_update = pyqtSignal(int)  # 更新进度条
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = True
        
    def run(self):
        while self.running:
            try:
                frame = self.processor.q.get(timeout=1.0)
                if frame is None:
                    break
                    
                # 处理帧
                processed_frame, result = self.processor.process_frame(frame)
                
                if processed_frame is not None:
                    self.frame_ready.emit(processed_frame)
                    
                if result:
                    self.detection_result.emit(result)
            except queue.Empty:
                # 队列为空，正常现象，无需报错
                continue
            except Exception as e:
                # 获取异常类型和消息
                error_msg = str(e).strip()
                if not error_msg:
                    error_msg = f"{type(e).__name__} (无错误消息)"

                # 发送带类型的信息
                self.status_update.emit(f"处理线程错误: {error_msg}")

                # 可选：打印详细堆栈到控制台（不影响GUI）
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()

        self.status_update.emit("图像处理线程已停止")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 数字识别系统")
        self.setGeometry(100, 100, 800, 600)

        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon('icon.png'))
        
        # 创建核心组件
        self.processor = None
        self.receiver = None
        self.detection_thread = None
        
        # 当前状态
        self.is_running = False
        self.excel_filename = ""
        self.logger = None
        
        # self.settings = QSettings("HuaMai", "YOLODetector")  # 使用注册表方式记录配置。两个参数分别是组织名和应用名
        self.settings = QSettings("config.ini", QSettings.IniFormat)  # 使用ini文件记录配置。
		
        self.init_ui()
        self.setup_logging()
        self.load_settings()  # 启动时加载设置

        # 创建日志信号
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self.update_log_display)

        # 设置日志 handler
        handler = QtHandler(self.log_emitter)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def update_log_display(self, msg):
        self.log_text.append(msg)

    def load_settings(self):
        """启动时加载上次保存的设置"""
        model_path = self.settings.value("model_path", "")
        ip = self.settings.value("ip", "192.168.31.1")
        port = self.settings.value("port", "8554")
        stream_name = self.settings.value("stream_name", "stream0")
        output_dir = self.settings.value("output_dir", "data")
        interval = int(self.settings.value("detection_interval", 2))
        min_value = int(self.settings.value("min_value", 0))
        max_value = int(self.settings.value("max_value", 999))
        display_checked = self.settings.value("display_check", True, type=bool)

        if model_path:
            self.model_path_input.setText(model_path)
        self.ip_input.setText(ip)
        self.port_input.setText(port)
        self.stream_name_input.setText(stream_name)
        self.output_dir_input.setText(output_dir)
        self.interval_spin.setValue(interval)
        self.min_value_spin.setValue(min_value)
        self.max_value_spin.setValue(max_value)
        self.display_check.setChecked(display_checked)

    def save_settings(self):
        """保存当前设置"""
        model_path = self.model_path_input.text().strip()
        ip = self.ip_input.text().strip()
        port = self.port_input.text().strip()
        stream_name = self.stream_name_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        if model_path:
            self.settings.setValue("model_path", model_path)
        if ip:
            self.settings.setValue("ip", ip)
        if port:
            self.settings.setValue("port", port)
        if stream_name:
            self.settings.setValue("stream_name", stream_name)
        if output_dir:
            self.settings.setValue("output_dir", output_dir)

        self.settings.setValue("detection_interval", self.interval_spin.value())
        self.settings.setValue("min_value", self.min_value_spin.value())
        self.settings.setValue("max_value", self.max_value_spin.value())
        self.settings.setValue("display_check", self.display_check.isChecked())

    def init_ui(self):
        """初始化用户界面"""
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("YOLO 数字识别系统")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 1. 配置选项卡
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "配置")
        
        # 2. 监控显示选项卡
        monitor_tab = self.create_monitor_tab()
        tab_widget.addTab(monitor_tab, "实时监控")
        
        # 3. 日志选项卡
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "日志")
        
        # 4. 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
        
        # 进度条
        # self.progress_bar = QProgressBar()
        # self.progress_bar.setMaximum(100)
        # self.progress_bar.setValue(0)
        # self.progress_bar.setTextVisible(True)
        # self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 定时器用于更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 每秒更新一次
        
    def create_config_tab(self):
        """创建配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 网络摄像头配置组
        camera_group = QGroupBox("摄像头配置")
        camera_layout = QFormLayout()
        
        self.ip_input = QLineEdit("192.168.31.1")
        self.ip_input.setPlaceholderText("输入摄像头IP地址")
        camera_layout.addRow("IP地址:", self.ip_input)
        
        self.port_input = QLineEdit("8554")
        camera_layout.addRow("端口:", self.port_input)
        
        self.stream_name_input = QLineEdit("stream0")
        camera_layout.addRow("流名称:", self.stream_name_input)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("选择YOLO模型文件 (.pt)")
        model_layout.addRow("模型路径:", self.model_path_input)
        
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self.browse_model)
        model_hbox = QHBoxLayout()
        model_hbox.addWidget(self.model_path_input)
        model_hbox.addWidget(browse_model_btn)
        model_layout.addRow("", model_hbox)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 检测参数组
        detect_group = QGroupBox("检测参数")
        detect_layout = QFormLayout()
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(0, 60)
        self.interval_spin.setValue(2)
        self.interval_spin.setSpecialValueText("每帧检测")
        detect_layout.addRow("检测间隔(秒):", self.interval_spin)
        
        self.min_value_spin = QSpinBox()
        self.min_value_spin.setRange(-9999, 9999)
        self.min_value_spin.setValue(0)
        detect_layout.addRow("最小值过滤:", self.min_value_spin)
        
        self.max_value_spin = QSpinBox()
        self.max_value_spin.setRange(-9999, 9999)
        self.max_value_spin.setValue(999)
        detect_layout.addRow("最大值过滤:", self.max_value_spin)
        
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)
        
        # 输出配置组
        output_group = QGroupBox("输出配置")
        output_layout = QFormLayout()
        
        self.output_dir_input = QLineEdit("data")
        self.output_dir_input.setPlaceholderText("输出文件目录")
        output_layout.addRow("输出目录:", self.output_dir_input)
        
        browse_output_btn = QPushButton("选择目录...")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        output_hbox = QHBoxLayout()
        output_hbox.addWidget(self.output_dir_input)
        output_hbox.addWidget(browse_output_btn)
        output_layout.addRow("", output_hbox)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 显示选项
        self.display_check = QCheckBox("显示实时画面窗口")
        self.display_check.setChecked(True)
        layout.addWidget(self.display_check)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("启动检测")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #27ae60; color: white;")
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #e74c3c; color: white;")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # 添加弹性空间
        layout.addStretch()
        
        return widget
        
    def create_monitor_tab(self):
        """创建现代化的监控显示选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ========================================
        # 1. 视频显示区域（主区域，可缩放）
        # ========================================
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.StyledPanel)
        video_frame.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("等待视频流...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #000000;
            color: #888888;
            font-size: 18px;
            font-weight: bold;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(480, 320)

        # 添加缩放策略提示（可选）
        # self.video_label.setScaledContents(True)  # 如果你想让图像自动缩放填充

        video_layout.addWidget(self.video_label)

        # 可选：添加控制按钮（截图、全屏）
        control_layout = QHBoxLayout()
        control_layout.addStretch()

        self.screenshot_btn = QPushButton("📷 截图")
        self.screenshot_btn.setFixedWidth(100)
        control_layout.addWidget(self.screenshot_btn)

        video_layout.addLayout(control_layout)
        layout.addWidget(video_frame, stretch=3)  # 主区域占 3 份

        # 添加分割线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line1)

        # ========================================
        # 2. 检测结果 + 统计信息（并排显示）
        # ========================================
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        # --- 左侧：检测结果（大字体）---
        result_group = QGroupBox("📌 当前检测值")
        result_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Digital-7", 28, QFont.Bold))  # 数码字体，可替换为 Arial
        self.result_label.setStyleSheet("""
            # background-color: #27ae60;
            color: white;
            border-radius: 10px;
            padding: 15px;
            font-weight: bold;
        """)
        self.result_label.setMinimumHeight(80)
        result_layout.addWidget(self.result_label)

        # 添加时间戳
        self.result_time_label = QLabel("未检测")
        self.result_time_label.setAlignment(Qt.AlignCenter)
        self.result_time_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        result_layout.addWidget(self.result_time_label)

        bottom_layout.addWidget(result_group, stretch=2)  # 占 2 份

        # --- 右侧：统计信息 ---
        stats_group = QGroupBox("📊 运行统计")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setHorizontalSpacing(20)
        stats_layout.setVerticalSpacing(10)

        # 定义标签和值
        self.frame_count_label = QLabel("0")
        self.detect_count_label = QLabel("0")
        self.record_count_label = QLabel("0")
        # self.fps_label = QLabel("0")

        # 设置值的样式
        for label in [self.frame_count_label, self.detect_count_label, self.record_count_label]:
            label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")

        # 网格布局：两行两列
        stats_layout.addWidget(QLabel("已处理帧数:"), 0, 0)
        stats_layout.addWidget(self.frame_count_label, 0, 1)
        stats_layout.addWidget(QLabel("检测次数:"), 0, 2)
        stats_layout.addWidget(self.detect_count_label, 0, 3)

        stats_layout.addWidget(QLabel("记录次数:"), 1, 0)
        stats_layout.addWidget(self.record_count_label, 1, 1)
        # stats_layout.addWidget(QLabel("实时 FPS:"), 1, 2)
        # stats_layout.addWidget(self.fps_label, 1, 3)

        bottom_layout.addWidget(stats_group, stretch=3)  # 占 3 份

        layout.addLayout(bottom_layout, stretch=1)

        return widget
        
    def create_log_tab(self):
        """创建日志选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_text)
        
        # 清除日志按钮
        clear_log_btn = QPushButton("清除日志")
        clear_log_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_log_btn)
        
        return widget
        
    def setup_logging(self):
        """初始化日志系统：生成文件名并配置日志"""
        import logging
        from datetime import datetime
        import os

        # === 1. 生成唯一文件名（保留你原来的逻辑）===
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"detected_numbers_{timestamp}"
        self.excel_filename = os.path.join("data", f"{base_filename}.xlsx")

        # 创建目录
        os.makedirs("data", exist_ok=True)
        os.makedirs("log", exist_ok=True)

        # === 2. 调用新的日志配置函数（使用 root logger）===
        self._configure_logging(self.excel_filename)

        # === 3. 现在可以安全使用 logging.info() 了 ===
        logging.info(f"日志系统已初始化: {self.excel_filename}")
        logging.info(f"Excel 文件将保存到: {self.excel_filename}")

        # GUI 显示（可选）
        self.append_log(f"日志系统已初始化: {os.path.basename(self.excel_filename)}")

    def _configure_logging(self, excel_filename):
        """配置日志处理器：文件、控制台、GUI"""
        import logging
        import os

        # 1. 获取 root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # 2. 清理旧 handlers（防止重复输出）
        for handler in logger.handlers[:]:
            if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                handler.close()
                logger.removeHandler(handler)

        # 3. 生成日志文件路径
        base_name = os.path.basename(excel_filename)
        log_filename = base_name.replace(".xlsx", ".txt")
        log_path = os.path.join("log", log_filename)

        # 4. 文件处理器（写入 log/xxx.txt）
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 5. 控制台处理器（可选）
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        # 注意：不要设置 logger.propagate = False

    def append_log(self, message):
        """向日志窗口添加消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        self.append_log("日志已清除")
        
    def browse_model(self):
        """浏览并选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLO模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
            self.save_settings()
            
    def browse_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_input.setText(dir_path)
            
    def start_detection(self):
        """启动检测"""
        if self.is_running:
            return

        self.save_settings()

        try:
            # 获取配置参数
            ip = self.ip_input.text().strip()
            port = self.port_input.text().strip()
            stream_name = self.stream_name_input.text().strip()
            model_path = self.model_path_input.text().strip()
            output_dir = self.output_dir_input.text().strip()
            interval = self.interval_spin.value()
            min_val = self.min_value_spin.value()
            max_val = self.max_value_spin.value()
            display = self.display_check.isChecked()
            
            # 验证输入
            if not ip:
                QMessageBox.warning(self, "警告", "请输入摄像头IP地址")
                return
                
            if not model_path or not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", "请指定有效的模型文件")
                return
                
            # 构建RTSP URL
            camera_url = f"rtsp://{ip}:{port}/{stream_name}"
            self.append_log(f"RTSP流地址: {camera_url}")
            
            # 创建输出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"detected_numbers_{timestamp}"
            self.excel_filename = os.path.join(output_dir, f"{base_filename}.xlsx")
            
            # 创建核心处理器
            self.processor = YOLOProcessor(
                model_path=model_path,
                interval=interval,
                min_val=min_val,
                max_val=max_val,
                display=display,
                excel_filename=self.excel_filename,
                logger=self.logger
            )
            
            # 创建接收器
            self.receiver = StreamReceiver(camera_url, self.processor.q, logger=self.logger)
            
            # 启动线程
            self.receiver.start()
            self.detection_thread = YOLODetectionThread(self.processor)
            self.detection_thread.frame_ready.connect(self.update_video_frame)
            self.detection_thread.detection_result.connect(self.update_detection_result)
            self.detection_thread.status_update.connect(self.append_log)
            # self.detection_thread.progress_update.connect(self.progress_bar.setValue)
            self.detection_thread.start()
            
            # 更新UI状态
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_bar.showMessage("检测运行中...")
            self.append_log("检测已启动")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动检测失败: {str(e)}")
            self.append_log(f"启动失败: {str(e)}")

    def stop_detection(self):
        """停止检测"""
        if not self.is_running:
            return

        self.is_running = False
        self.processor.running = False
        self.receiver.running = False

        # 等待线程结束
        if hasattr(self.receiver, 'is_alive') and callable(getattr(self.receiver, 'is_alive')):
            if self.receiver.is_alive():
                self.receiver.join(timeout=3)
        else:
            # 如果receiver是QThread或没有is_alive方法
            pass

        # 修正：使用 isRunning() 而不是 is_alive()
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait(3000)  # 3秒超时

        self.processor.write_to_excel()

        # 清理资源
        if hasattr(self.processor, 'model'):
            del self.processor.model

        # 更新UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("检测已停止")
        self.append_log("检测已停止")

        # 显示最终统计
        if hasattr(self.processor, 'frame_count'):
            self.append_log(f"总计处理帧数: {self.processor.frame_count}")
            self.append_log(f"检测次数: {self.processor.detection_count}")
            self.append_log(f"记录次数: {self.processor.record_count}")
            
    def update_video_frame(self, frame):
        """更新视频显示"""
        if frame is None:
            return
            
        # 转换为QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 调整大小以适应显示区域
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_detection_result(self, result):
        """更新检测结果显示"""
        detected_num = result.get('Detected Number', 'N/A')
        self.result_label.setText(f"检测到: {detected_num}")
        
        # 更新统计
        if hasattr(self.processor, 'frame_count'):
            self.frame_count_label.setText(str(self.processor.frame_count))
        if hasattr(self.processor, 'detection_count'):
            self.detect_count_label.setText(str(self.processor.detection_count))
        if hasattr(self.processor, 'record_count'):
            self.record_count_label.setText(str(self.processor.record_count))
            
    def update_status(self):
        """定期更新状态"""
        if self.is_running:
            # 可以在这里添加更多实时状态信息
            pass
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_running:
            reply = QMessageBox.question(
                self, '确认',
                "检测正在运行，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_detection()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# 启动应用
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())