
import matplotlib.pyplot as plt
import gzip, os, sys
import numpy as np
from scipy.stats import multivariate_normal

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
#     print("Downloading %s" % filename)
#     urlretrieve(source + filename, filename)

# Invokes download() if necessary, then reads in images 加载图像样本
def load_mnist_images(filename):
    # if not os.path.exists(filename):
    #     download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data
# 加载标签
def load_mnist_labels(filename): 
    # if not os.path.exists(filename):
    #     download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

## Load the training set
#path = 'C:\Courses\Edx\Current\USCD ML\Week3\\'
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

## Load the testing set
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

## Define a function that displays a digit given its vector representation
# def show_digit(x, label):
#     plt.axis('off')
#     plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
#     plt.title('Label ' + str(label))
#     #plt.show()
#     #return

# ## Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
# def vis_image(index, dataset="train"):
#     if(dataset=="train"):
#         label = train_labels[index]
#         show_digit(train_data[index,], label)
#     else:
#         label = test_labels[index]
#         show_digit(test_data[index,], label)
#     #print("Label " + str(label))
#     plt.show()
#     return

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     #show_digit(train_data[i,], train_labels[i])
#     show_digit(test_data[i,], test_labels[i])
# plt.tight_layout()
# plt.show()

# test_data1 = test_data[500,:]
# test_labels1 = test_labels[500]
# plt.figure
# show_digit(test_data1, test_labels1)
# plt.tight_layout()
# plt.show()

# #########################  下面的程序是分别是用不同分类器实现数字分类算法 ##############################
# ##### knn 分类器
# import time
# from sklearn.neighbors import BallTree
# ## Build nearest neighbor structure on training data
# t_before = time.time()
# ball_tree = BallTree(train_data[3000:4000,:])
# t_after = time.time()
# # Compute training time
# t_training = t_after - t_before
# print("Time to build data structure (seconds): ", t_training)
# ## Get nearest neighbor predictions on testing data
# test_data1=test_data[500,:]
# test_labels1=test_labels[500]
#
# t_before = time.time()
# test_data1 = np.expand_dims(test_data1,0)
# test_neighbors = ball_tree.query(test_data1, k=100, return_distance=False)
# test_neighbors = np.squeeze(test_neighbors)
# # test_neighbors = np.squeeze(ball_tree.query(test_data1, k=1, return_distance=False))
# test_predictions = train_labels[test_neighbors]
# t_after = time.time()
# ## Compute testing time
# t_testing = t_after - t_before
# print("Time to classify test set (seconds): ", t_testing)
# # evaluate the classifier
# t_accuracy = sum(test_predictions == test_labels1) / 100 #float(len(test_labels1))
# # t_accuracy
# print("accuracy: ", t_accuracy)
# 0.96909999999999996
#
# import pandas as pd
# import seaborn as sn
# from sklearn import metrics
# cm = metrics.confusion_matrix(test_labels,test_predictions)
# df_cm = pd.DataFrame(cm, range(10), range(10))
# sn.set(font_scale=1.2)# for label size
# sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt="g")

### Bayes 分类器
# def display_char(image):
#     plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#     plt.axis('off')
#     plt.show()
#
# def fit_generative_model(x,y):
#     k = 10  # labels 0,1,...,k-1
#     d = (x.shape)[1]  # number of features
#     mu = np.zeros((k,d))
#     sigma = np.zeros((k,d,d))
#     pi = np.zeros(k)
#     c = 3500 #10000 #1000 #100 #10 #0.1 #1e9
#     for label in range(k):
#         indices = (y == label)
#         pi[label] = sum(indices) / float(len(y))
#         mu[label] = np.mean(x[indices,:], axis=0)
#         sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1) + c*np.eye(d)
#     return mu, sigma, pi
# mu, sigma, pi = fit_generative_model(train_data, train_labels)
# display_char(mu[0])
# display_char(mu[1])
# display_char(mu[2])
#
# test_data1=test_data[0:5000,:]
# test_labels1=test_labels[0:5000]
# k = 10
# score = np.zeros((len(test_labels1),k))
# for label in range(0,k):
#     rv = multivariate_normal(mean=mu[label], cov=sigma[label])
#     for i in range(0,len(test_labels1)):
#        score[i,label] = np.log(pi[label]) + rv.logpdf(test_data1[i,:])
# test_predictions = np.argmax(score, axis=1)
# # Finally, tally up score
# errors = np.sum(test_predictions != test_labels1)
# print("The generative model makes " + str(errors) + " errors out of 20")
# t_accuracy = sum(test_predictions == test_labels1) / float(len(test_labels1))
# print("t_accuracy " ,t_accuracy )

# ### SVM 分类器
# from sklearn import svm
# # C = 1
# #for C in [.01,.1,1.,10.,100.]:
# # clf = SVC(C=C, kernel='poly', degree=2)

# clf = svm.SVC() # 定义svm分类器
# clf.fit(train_data[3000:4000,:],train_labels[3000:4000]) # 训练

# # test_data1 = np.expand_dims(test_data[0,:],0)
# # testpredict = clf.predict(test_data1)
# # plt.figure
# # show_digit(test_data[0,:], test_labels[0])
# # plt.tight_layout()
# # plt.show()
# # print( clf.score(test_data,test_labels))

# import pandas as pd
# import seaborn as sn
# from sklearn import metrics
# test_predictions = clf.predict(test_data)  # 所有测试数据预测
# cm=metrics.confusion_matrix(test_labels,test_predictions) # 计算混淆矩阵
# df_cm = pd.DataFrame(cm, range(10), range(10))
# plt.figure()
# sn.set(font_scale=1.2)#for label size
# sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt="g") #, cmap='viridis')# font size
# plt.show()

# # wrong_indices = test_predictions != test_labels
# # wrong_digits, wrong_preds, correct_labs = test_data[wrong_indices], test_predictions[wrong_indices], test_labels[wrong_indices]
# # print(len(wrong_pred))
# # plt.title('predicted: ' + str(wrong_preds[1]) + ', actual: ' + str(correct_labs[1]))
# # display_char(wrong_digits[1])



###### ANN 分类器
# import pandas as pd
# import seaborn as sn
# from sklearn import metrics
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(100,),
#                     activation='logistic', solver='adam',
#                     learning_rate_init = 0.0001, max_iter=2000)
#
# print(clf)
# clf.fit(train_data[3000:4000,:],train_labels[3000:4000])
# test_predictions = clf.predict(test_data)
#
#
#
# cm=metrics.confusion_matrix(test_labels,test_predictions)
# df_cm = pd.DataFrame(cm, range(10), range(10))
# plt.figure()
# sn.set(font_scale=1.2)#for label size
# sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt="g") #, cmap='viridis')# font size
# plt.show()
# t_accuracy = sum(test_predictions == test_labels) / float(len(test_labels))
# print(t_accuracy)