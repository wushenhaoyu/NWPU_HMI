# -*- coding: utf-8 -*-
import wave
import time
import librosa
import numpy as np
import pvrecorder
import pyaudio
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QIODevice, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaRecorder
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

list = ["Takeoff", "Landing", "Advance", "Retreat", "Rise"]

class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.recorder_thread = None
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 513)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(600, 280, 321, 121))
        self.toolButton.clicked.connect(self.recordAudio)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(36)
        self.toolButton.setFont(font)
        self.toolButton.setObjectName("toolButton")
        self.timer = QTimer(self.centralwidget)
        self.timer.timeout.connect(self.update_progress)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(600, 20, 291, 71))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(28)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 540, 391))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(770, 90, 281, 101))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(36)
        self.label_3.setFont(font)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 422, 771, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setMaximum(1000)
        self.progressBar.setValue(0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.menu.addAction(self.action)
        self.menubar.addAction(self.menu.menuAction())
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.model = MFCC_CNN()
        model_weights_path = 'best_model_weights.pth'
        self.model.load_state_dict(torch.load(model_weights_path, weights_only=True))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
        self.model.eval()




    def update_progress(self, value):
        self.progressBar.setValue(value)
    def recordAudio(self):

        self.progressBar.setValue(0)
        self.recorder_thread = RecordingThread(self.centralwidget)
        self.recorder_thread.progress_updated.connect(self.update_progress)
        self.recorder_thread.recording_finished.connect(self.on_recording_finished)
        self.recorder_thread.start()

    def on_recording_finished(self):
        # 加载音频文件并提取 MFCC
        data, sr = librosa.load('output.wav', sr=None)
        data = librosa.effects.preemphasis(data)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T

        # 清除之前的图像
        self.figure.clear()

        # 绘制 MFCC
        ax = self.figure.add_subplot(111)
        img = librosa.display.specshow(mfcc.T, sr=sr, hop_length=512, x_axis='time', ax=ax)
        self.figure.colorbar(img, ax=ax)
        ax.set(title='MFCC visualization of Sample')

        # 更新画布
        self.canvas.draw()

        # 抓取画布内容并设置为标签的 pixmap
        canvas_image = self.canvas.grab()
        pixmap = QPixmap(canvas_image)
        self.label_2.setPixmap(pixmap)

        # 预处理 MFCC 数据
        mfcc = zero_pad(mfcc)
        mfcc = torch.tensor(mfcc).unsqueeze(0).to(self.device)

        # 使用模型进行预测
        prediction = self.model(mfcc)
        self.label_3.setText(list[torch.argmax(prediction)])

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.toolButton.setText(_translate("MainWindow", "开始录音"))
        self.label.setText(_translate("MainWindow", "预测结果："))
        self.progressBar.setFormat(_translate("MainWindow", "%p%"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.action.setText(_translate("MainWindow", "导入"))

class MFCC_CNN(nn.Module):
    def __init__(self):
        super(MFCC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 54 * 3, 2048)
        self.fc2 = nn.Linear(2048, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)  # 应用Dropout层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def zero_pad(feature, max_length=216):
    # 计算需要填充的长度
    difference = max_length - feature.shape[0]

    # 对单个特征进行零填充
    padded_feature = np.pad(feature, ((0, difference), (0, 0)), "constant")

    return padded_feature

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class RecordingThread(QThread):
    progress_updated = pyqtSignal(int)
    recording_finished = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False

    def run(self):
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024)


        self.recording = True
        duration_seconds = 2        # 录音 2 秒
        start_time = time.time()
        total_frames = int(44100 * duration_seconds / 1024)
        frame_duration_ms = 1024 / 44100 * 1000  # 每帧持续时间 (毫秒),约为23ms
        max_prograss = 1000     #进度条最大时间
        for i in range(total_frames):
            if not self.recording:
                break
            data = self.stream.read(1024)
            self.frames.append(data)
            elapsed_time = time.time() - start_time
            progress = int((elapsed_time / duration_seconds) * max_prograss)
            progress = min(progress, max_prograss)
            self.progress_updated.emit(progress)
            #self.msleep(int(frame_duration_ms))  # 模拟每 23 ms 更新一次进度条

        if self.recording:
            # 停止录音
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()

        # 保存录音文件
        with wave.open('output.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))

        # 关闭 PyAudio
        self.audio.terminate()

        self.recording_finished.emit()



