
import os
import sys
import numpy
import librosa
import pandas as pd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


audio_path = 'train/'
sample_path = 'train/029_1_1.wav'


# 绘制样本音频波形图，声谱图，梅尔声谱图,梅尔系数图
data, sr = librosa.load(sample_path)     #   sr->sample rating,采样率
data = librosa.effects.preemphasis(data)       #预加重处理


def wave(data,sr):
    plt.title('Wave of Sample')
    librosa.display.waveshow(data,sr=sr)
    plt.grid(True)


def spectrogram(data,sr):
    S = librosa.stft(data)
    S_db = librosa.amplitude_to_db(abs(S))

    plt.title('Spectrogram of Sample')
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


def mel_spectrogram(data,sr):
    S_Mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)   #128个梅尔滤波器
    S_Mel_db = librosa.amplitude_to_db(abs(S_Mel))
    plt.title('Mel_Spectrogram of Sample')
    librosa.display.specshow(S_Mel_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


# 可视化MFCC特征
def mfcc_visualization(sample_path):
    y, sr = librosa.load(sample_path)
    y = librosa.effects.preemphasis(y)  # 预加重处理
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512).T
    '''
       取每一帧长度为25ms，帧间隔10ms，采样率sr为22050hz
       每一帧采样数22050*0.025=551
       每一帧包含551个点，作FFT时可选n_fft=512，作512次FFT
       '''
    librosa.display.specshow(mfcc_features.T, sr=sr, hop_length=512, x_axis='time')
    plt.colorbar()
    plt.title('MFCC visualization of Sample')


def draw_figures():
    plt.subplot(2,2,1)
    wave(data,sr)
    plt.subplot(2,2,2)
    spectrogram(data,sr)
    plt.subplot(2,2,3)
    mel_spectrogram(data,sr)
    plt.subplot(2,2,4)
    mfcc_visualization(sample_path)

    plt.suptitle(f"Sample_path:{os.path.basename(sample_path)}")
    plt.tight_layout()
    plt.show()
#   *******************************
#draw_figures()
#   *******************************


students_num = ["029", "033", "039", "045", "049", "068", "914", "918", "919", "934", "970"]
#orders_num = ["起飞", "降落", "前进", "后退", "升高"]
orders_num = ["Takeoff", "Landing", "Advance", "Retreat", "Rise"]
#orders_num = ["起飞-1", "降落-2", "前进-3", "后退-4", "升高-5"]
#orders_num = ["Takeoff-1", "Landing-2", "Advance-3", "Retreat-4", "Rise-5"]
repeat_num = ['1', '2', '3', '4']
number_of_mfcc_features = 13


# 提取MFCC特征
def mfcc_extraction(path, students_num, orders_num, repeat_num):
    labels = []
    mean_features = []  #平均值
    std_features = []   #标准差

    for stu_num in students_num:
        for index, order in enumerate(orders_num):
            for repeat in repeat_num:
                file_path = path + stu_num + '_' + str(index+1) + '_' + repeat + ".wav"
                if os.path.exists(file_path):
                    x, sr = librosa.load(file_path)
                    x = librosa.effects.preemphasis(x)  # 预加重处理
                    mean_mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T, axis=0)
                    std_mfccs = np.std(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=number_of_mfcc_features).T, axis=0)

                    mean_features.append(mean_mfccs)
                    std_features.append(std_mfccs)

                    labels.append(order)
                else:
                    pass
    return mean_features, std_features, labels


mfcc_features_and_labels = mfcc_extraction(audio_path, students_num, orders_num, repeat_num)
mean_mfcc_features, std_mfcc_features, mfcc_labels = mfcc_features_and_labels
# MFCC特征整合
mfcc_features = numpy.concatenate((mean_mfcc_features, std_mfcc_features,), axis=1)
#print(mfcc_features.shape)


# PCA分析，选择主要特征
from sklearn.decomposition import PCA
'''
pca = PCA().fit(mfcc_features)
#绘制方差解释图
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))  #cumsum 计算方差解释的累计和
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #方差解释的累计和
plt.title('MFCC Explained Variance')
plt.show()
'''
pca = PCA(n_components=20)  #根据方差解释图，选择20个主要特征
mfcc_features = pca.fit_transform(mfcc_features)    #获取主要成分的数据
#print(mfcc_features.shape)


# 数据标准化
from sklearn.preprocessing import StandardScaler
data = mfcc_features
scaler = StandardScaler()
mfcc_features = scaler.fit_transform(data)


# 过采样解决类别不均衡
from imblearn.over_sampling import RandomOverSampler
'''
dictionary = {}
for i in mfcc_labels:
    dictionary[i] = dictionary.get(i,0) + 1
print(dictionary)
#>>>{'Takeoff': 23, 'Landing': 23, 'Advance': 24, 'Retreat': 22, 'Rise': 20}
'''
ros = RandomOverSampler(random_state=42)
new_mfcc_features,  mfcc_labels = ros.fit_resample(mfcc_features,  mfcc_labels)
'''
dictionary_2 = {}
for i in mfcc_labels:
    dictionary_2[i] = dictionary_2.get(i,0) + 1
print(dictionary_2)
#>>>{'Takeoff': 24, 'Landing': 24, 'Advance': 24, 'Retreat': 24, 'Rise': 24}
'''


# SVM分类器分类
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 划分训练集，测试集
X_train, X_test, y_train, y_test = train_test_split(new_mfcc_features, mfcc_labels, test_size=0.30,
                                                    random_state=42, stratify=mfcc_labels, shuffle=True)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"accuracy:{accuracy_score(y_test, y_pred):.10f}")

# GridSearch,网格搜索，优化超参数
from sklearn.model_selection import GridSearchCV

# 定义参数范围，C惩罚系数，gamma时rbf核函数的参数，kernel核函数（线性，多项式，高斯）
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf']}

grid = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy', n_jobs=-1,)

# 带入模型，输出最佳cv=10折交叉验证准确率，最佳的模型参数
grid.fit(X_train, y_train)
print(f"GridSearch Cross verification accuracy：{grid.best_score_:.10f}")
print(grid.best_params_)
print(grid.best_estimator_)

# 用网格搜索后的参数模型
from sklearn.metrics import classification_report, confusion_matrix
clf = grid.best_estimator_
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred))
print()
print(f"GridSearch accuracy:{accuracy_score(y_test, y_pred):.10f}")

# 混淆矩阵
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
plt.gcf().subplots_adjust(left=0.25,right=0.95,bottom=0.19,top=0.8) #设置图像边界
plt.imshow(confmat,cmap='Blues',aspect='auto')
ax.xaxis.set_ticks_position('bottom')
# 填入数据
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],
                va='center',ha='center')

plt.title('Confusion Matrix with labels')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(ticks=np.arange(len(orders_num)),labels=orders_num,horizontalalignment='center')
plt.yticks(ticks=np.arange(len(orders_num)),labels=orders_num,verticalalignment='center')
plt.tick_params(axis='both',which='minor',color='blue',pad=5,labelsize=4,direction='out')
plt.colorbar(pad=0.09,orientation='vertical',extend='both',shrink=0.8,location='right')

plt.tight_layout()
plt.show()

