# coding=utf-8
import os, time, copy
from glob import glob
import numpy as np
import pandas as pd
from skimage.color import gray2rgb
from random import seed, shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,plot_roc_curve,auc,roc_auc_score,roc_curve,accuracy_score,f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score
from tf_geometric.utils import tf_utils
import tf_geometric as tfg

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

# 对数据集进行筛选，对每个结节选取指定数量的切片
def cutDF(data, slices_filter):
    # slices_filter: 中心切片上下取几张，该计数包含中心切片
    data = data.loc[data["slices"] >= slices_filter,]  # 挑选总切片数[slices_filter]的结节
    data = data.loc[(data["Z"] <= data["Zmed"] + slices_filter//2)
                  &(data["Z"] >= data["Zmed"] - slices_filter//2),].reset_index(drop=True)  # 每个结节取中心上下共[slices_filter]张
    return data

# 数据集划分训练集和测试集
def splitDF(data, col_name, test_size=0.3, reset=True, seed=123, shuffle=False):
    # 全部结节ID列表
    nIDs = data["nID"].unique().tolist()
    # 结节ID列表划分为训练集和测试集
    train_nIDs, test_nIDs = train_test_split(nIDs, test_size=test_size, random_state=seed, shuffle=shuffle)
    # 得到两个数据集
    train_dat = data.loc[data[col_name].isin(train_nIDs), ]  # 训练集
    test_dat = data.loc[data[col_name].isin(test_nIDs),]  # 测试集
    if reset:  # 判断是在划分训练集和测试集，还是在划分训练集和验证集
        train_dat = train_dat.reset_index(drop=True)  # 训练集
        test_dat = test_dat.reset_index(drop=True)  # 测试集
    # 提取索引，用于模型训练
    train_index = train_dat.index.tolist()  # 训练集索引
    test_index = test_dat.index.tolist()  # 测试集索引
    # 返回训练集表、测试集表、训练集表索引、测试集表索引
    return train_dat, test_dat, train_index, test_index

# 按病人 数据集划分训练集和测试集
def psplitDF(data, col_name, test_size=0.2, reset=True, seed=123, shuffle=False):
    # 全部结节ID列表
    IDs = data["ID"].unique().tolist()
    # 结节ID列表划分为训练集和测试集
    train_IDs, test_IDs = train_test_split(IDs, test_size=test_size, random_state=seed, shuffle=shuffle)
    #X=len(IDs)
    #kf = KFold(n_splits=5,shuffle=False)
    #for train_index , test_index in kf.split(X):
    #    print('train_index:%s , test_index: %s ' %(train_index,test_index))
    # 得到两个数据集
    train_dat = data.loc[data[col_name].isin(train_IDs), ]  # 训练集
    test_dat = data.loc[data[col_name].isin(test_IDs),]  # 测试集
    if reset:  # 判断是在划分训练集和测试集，还是在划分训练集和验证集
        train_dat = train_dat.reset_index(drop=True)  # 训练集
        test_dat = test_dat.reset_index(drop=True)  # 测试集
    # 提取索引，用于模型训练
    train_index = train_dat.index.tolist()  # 训练集索引
    test_index = test_dat.index.tolist()  # 测试集索引
    # 返回训练集表、测试集表、训练集表索引、测试集表索引
    return train_dat, test_dat, train_index, test_index


# 计算各数据子集中0-1分类占比，观察样本平衡性
def calRate(data, index=None):
    # 是否通过索引进行筛选
    if index is None:
        n = data.shape[0]
    else:
        n = len(index)
        data = data.loc[index, ]  #获取指定索引的数据集
    # 结节数量
    nn = len(data["nID"].unique())
    pp = len(data["ID"].unique())
    rate1 = round(data["flag"].sum() / n * 100, 2)  # 计算数据子集中微乳头切片占比
    rate2 = round(data.grouby("nID")["flag"].mean().mean() * 100, 2)  # 计算数据子集中微乳头结节占比
    return f"切片数: {n}，微乳头切片占比: {rate1}%，结节数: {nn}，微乳头结节占比：{rate2}%，病人数: {pp}"


# 按病人计算各数据子集中0-1分类占比，观察样本平衡性
def pcalRate(data, index=None):
    # 是否通过索引进行筛选
    if index is None:
        n = data.shape[0]
    else:
        n = len(index)
        data = data.loc[index, ]#获取指定索引的数据集
    # 结节数量
    nn = len(data["nID"].unique())
    pp=  len(data["ID"].unique())
    rate1 = round(data["flag"].sum() / n * 100, 2)  # 计算数据子集中微乳头切片占比
    rate2 = round(data.groupby("ID")["flag"].mean().mean() * 100, 2)  # 计算数据子集中微乳头结节占比
    return f"切片数: {n}，微乳头切片占比: {rate1}%，结节数: {nn}, 微乳头结节占比：{rate2}%，病人数: {pp}"


# 构建图的双向边，链状图
def createLinearEdges(data, col_name):
    edges = pd.DataFrame(columns=["source", "target"])  # 初始化
    crits = data[col_name].unique().tolist()  # 提取结节的唯一编码
    for nID in crits:
        tmp = data.loc[data[col_name]==nID,]  # 提取对应nID的结节
        # 结节任意相邻的两张切片互相指向
        # 第一条边，从较低的Z指向较高的Z
        source = tmp.iloc[0:tmp.shape[0]-1,].index  # 边起始索引
        target = tmp.iloc[0:tmp.shape[0]-1,].index+1  # 边终点索引
        edges = edges.append(pd.DataFrame({"source":source,"target":target}))
        # 第二条边，从较高的Z指向较低的Z
        source = tmp.iloc[1:tmp.shape[0],].index  # 边起始索引
        target = tmp.iloc[1:tmp.shape[0],].index-1  # 边终点索引
        edges = edges.append(pd.DataFrame({"source":source,"target":target}))
    edges = edges.reset_index(drop=True)  # 重置索引
    return edges

# 构建图的双向边，星形图
def createStarEdges(data, col_name, slices_filter=None):
#     idx_start = slices_filter // 2  # 中心切片的起始索引
#     idx_range = slices_filter + 1  # 结节的切片数
#     cens = pd.DataFrame({"node": data.index[idx_start::idx_range], "key": data[col_name][idx_start::idx_range]})  # 中心切片索引
    tmp = data.loc[data["Z"]==data["Zmed"], "nID"]
    source = pd.DataFrame({"node": tmp.index.tolist(), "key": tmp.tolist()})  # 中心切片索引
    target = pd.DataFrame({"node": data.index, "key": data[col_name]})  # 全部索引
    # 以中心切片为起点，结节所有切片为终点，构建单向边
    edges1 = pd.merge(source, target, on="key", how="left").rename(columns={"node_x": "source", "node_y": "target"})
    # 以结节所有切片为起点，中心切片为终点，构建单向边
    edges2 = pd.merge(target, source, on="key", how="left").rename(columns={"node_x": "source", "node_y": "target"})
    # 单向边合并为双向边
    edges = pd.concat([edges1, edges2], axis=0)
    # 过滤中心切片指向自身的边
    edges = edges.loc[edges["source"] != edges["target"], ["source", "target"]].reset_index(drop=True)
    return edges

# 构建图的双向边，全连接图
def creatFullEdges(data, col_name):
    source = target = pd.DataFrame({"node": data.index.tolist(), "key": data[col_name]})  # 全部索引
    # 以中心切片为起点，结节所有切片为终点，构建单向边
    edges1 = pd.merge(source, target, on="key").rename(columns={"node_x": "source", "node_y": "target"})
    # 以结节所有切片为起点，中心切片为终点，构建单向边
    edges2 = pd.merge(target, source, on="key").rename(columns={"node_x": "source", "node_y": "target"})
    # 单向边合并为双向边
    edges = pd.concat([edges1, edges2], axis=0)
    # 过滤中心切片指向自身的边
    edges = edges.loc[edges["source"] != edges["target"], ["source", "target"]].reset_index(drop=True)
    return edges


# 构建图
def createGraph(data, col_name = "nID", edge_type = "linear"):
    if edge_type == "star":
        edges = createStarEdges(data, col_name)  # 星形图
    elif edge_type == "full":
        edges = creatFullEdges(data, col_name)  # 星形图
    else:
        edges = createLinearEdges(data, col_name)  # 链状图
    # 构图
    graph = tfg.Graph(
        x = np.array(data.iloc[:,-512:]), 
        edge_index = [edges['source'].tolist(),
                      edges['target'].tolist()],
        y = np.array(data['flag'])
    )
    return graph, edges

# 创建数据生成器的核心函数
def data_generator(index_list, batch_size=1, n_labels=1, labels=None, shuffle_index_list=True):
    orig_index_list = index_list.copy()  # 备份原始文件列表
    while True:
        # 准备数据
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            seed(123)
            shuffle(index_list)
        while len(index_list) > 0:
           # print('list长度：',len(index_list),len(x_list))
            index = index_list.pop()
            data, truth = get_data_from_file(index)
            x_list.append(data)
            y_list.append(truth)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()

# 读取图像和标签
def get_data_from_file(index):
    x = np.load(index[0])  # 读取npy，[60,60]
    x = gray2rgb(x) # 灰度图转RGB，[60,60,3]GCN
    #x = np.repeat(x[..., np.newaxis], 1, -1)3dcnn
#     x = np.repeat(x[..., np.newaxis], 3, -1)  # 单通道复制为3通道，[60,60,3]
    y = index[1]  # 0-1标签
    return x, y



##3dresnet
#3d数据生成器
def data_generator3d(index_list, batch_size=1, n_labels=1, labels=None, shuffle_index_list=True):
    orig_index_list = index_list.copy()  # 备份原始文件列表
    while True:
        # 准备数据
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            seed(123)
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            data, truth = get_data_from_file3d(index)
            x_list.append(data)
            y_list.append(truth)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()
#获取图像数据和标签
def get_data_from_file3d(index):
    image_path, label = index  # 获取图像路径和标签
    x = np.load(image_path)  # 读取npy，[60, 60]
    x = gray2rgb(x)  # 灰度图转RGB，[60, 60, 3]GCN
    # x = np.repeat(x[..., np.newaxis], 1, -1)3dcnn
    # x = np.repeat(x[..., np.newaxis], 3, -1)  # 单通道复制为3通道，[60, 60, 3]
    return x, label

# 标签类别转换
def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    #print(x.shape,y.shape)
    return x, y

# 处理多类别标签
def get_multi_class_labels(data, n_labels, labels=None):
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y

# 根据样本量和batch size计算训练的步数
def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

# 自定义callback
def get_callbacks(model_file, initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None,tensorboard_callback = None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file,monitor='val_binary_accuracy', save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_binary_accuracy',verbose=verbosity, patience=early_stopping_patience))
    #if tensorboard_callback:
        #callbacks.append(tensorboard_callback)
    return callbacks


# 计算metrics
def calMatrix(y_true, y_pred):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()#拉成一维
    # Accuracy
    print("Accuracy:{:.4f}".format(accuracy_score(y_true, y_pred)))
    # 特异性：TN / N
    specificity = tn / (tn+fp)
    # 敏感度：TP / P
    sensitivity= tp/(tp+fn)
    # Positive predictive value PPV = TP / (TP + FP)
    ppv = tp / (tp+fp)
    # Negative predictive value NPV = TN / (TN + FN)
    npv = tn / (tn+fn)
    print("Specificity:{:.4f}".format(specificity))
    print("Sensitivity:{:.4f}".format(sensitivity))
    print("PPV:{:.4f}".format(ppv))
    print("NPV:{:.4f}".format(npv))
    # F1-score
    print("F1-Score:{:.4f}".format(f1_score(y_true,y_pred)))


# 计算auc
def calAUC(y_true, y_value):
    # Accuracy
    preds = np.argmax(y_value, axis=1)
    print("Accuracy:{:.4f}".format(accuracy_score(y_true, preds)))
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_value[:,1])
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    # 最佳阈值
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    bestThresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return bestThresh


# 随便复制的一个绘制混淆矩阵的代码
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

#     fig, ax = plt.subplots()
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


import matplotlib.pyplot as plt
import matplotlib as mpl

font_name = "WenQuanYi Zen Hei"
mpl.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
mpl.rcParams['axes.unicode_minus']=False




# 综合评估函数，计算混淆矩阵并绘图
def compreEval(y_true, y_pred, y_value, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel() ##ravel数组维度拉成一维数组
    # Accuracy
    acc1 = accuracy_score(y_true, y_pred)
#     print("Accuracy:{:.4f}".format(acc1))
    # 特异性：TN / N
    specificity = tn / (tn+fp)
    # 敏感度：TP / P
    sensitivity= tp/(tp+fn)
    # Positive predictive value PPV = TP / (TP + FP)
    ppv = tp / (tp+fp)
    # Negative predictive value NPV = TN / (TN + FN)
    npv = tn / (tn+fn)
#     print("Specificity:{:.4f}".format(specificity))
#     print("Sensitivity:{:.4f}".format(sensitivity))
#     print("PPV:{:.4f}".format(ppv))
#     print("NPV:{:.4f}".format(npv))
    # F1-score
    f1 = f1_score(y_true,y_pred) ##结果是类别1的score
#     print("F1-Score:{:.4f}".format(f1))
    
    # 计算AUC
    # Accuracy
    preds = np.argmax(y_value, axis=1)
    acc2 = accuracy_score(y_true, preds)
#     print("Accuracy:{:.4f}".format(acc2))
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_value[:,1])
    
    roc_auc = auc(fpr, tpr)
   
    # 计算AUC值
       
#     print("Area under the ROC curve : %f" % roc_auc)
    # 最佳阈值
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    bestThresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]] ##argsort数组中的元素从小到大排序后的索引数组值
    
    
    # 绘制混淆矩阵
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    classes=['Low','High']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
        print('Confusion matrix, without normalization')

    print(cm)

#     fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
          #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 绘制ROC曲线
    ax = fig.add_subplot(122)
    ax.plot(fpr, tpr, label='ROC')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    
    # 整合评估指标
    metrics = pd.DataFrame({"ACC_pred": [acc1], "Specificity": [specificity], "Sensitivity": [sensitivity],
                            "PPV": [ppv], "NPV": [npv], "F1-Score": [f1], "ACC_argmax": [acc2], "AUC": [roc_auc]})
    
    fig.tight_layout()
    return metrics,bestThresh

# 以结节为单位进行评估
def metricsOnNodule(data, indexs, y_pred, y_value, class_names, pthresh=0.5):
    if indexs is None:
        noduleData = data.loc[:, ["nID", "ID", "flag"]].copy()  # copy测试集原数据
    else:
        noduleData = data.loc[indexs, ["nID", "ID", "flag"]].copy()  # copy测试集原数据
    noduleData["y_pred"] = y_pred
    noduleData[["0_value","1_value"]] = y_value
    noduleData = noduleData.groupby("nID").mean()  # 平均法
    noduleData[["ID","flag"]] = noduleData[["ID", "flag"]].astype(int)  # 保持数据类型
    noduleData["voting_pred"] = noduleData["y_pred"].apply(lambda x: 1 if x > pthresh else 0)  # 投票法得到预测标签，90%阈值
    return compreEval(noduleData["flag"], noduleData["voting_pred"], np.asarray(noduleData[["0_value","1_value"]]), classes=class_names, normalize=False)


# 综合评估函数，计算混淆矩阵并绘图
def compreEval_tf(y_true, y_pred, y_value, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel() ##ravel数组维度拉成一维数组
    # Accuracy
    acc1 = accuracy_score(y_true, y_pred)
#     print("Accuracy:{:.4f}".format(acc1))
    # 特异性：TN / N
    specificity = tn / (tn+fp)
    # 敏感度：TP / P
    sensitivity= tp/(tp+fn)
    # Positive predictive value PPV = TP / (TP + FP)
    ppv = tp / (tp+fp)
    # Negative predictive value NPV = TN / (TN + FN)
    npv = tn / (tn+fn)
#     print("Specificity:{:.4f}".format(specificity))
#     print("Sensitivity:{:.4f}".format(sensitivity))
#     print("PPV:{:.4f}".format(ppv))
#     print("NPV:{:.4f}".format(npv))
    # F1-score
    f1 = f1_score(y_true,y_pred) ##结果是类别1的score
#     print("F1-Score:{:.4f}".format(f1))
    
    # 计算AUC
    # Accuracy
    #preds = np.argmax(y_value, axis=1)
    acc2 = accuracy_score(y_true, pred)
#     print("Accuracy:{:.4f}".format(acc2))
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_value[:,1])
    roc_auc = auc(fpr, tpr)
#     print("Area under the ROC curve : %f" % roc_auc)
    # 最佳阈值
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    bestThresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]] ##argsort数组中的元素从小到大排序后的索引数组值
    
    
    # 绘制混淆矩阵
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes=['低危型结节','高危型结节']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
        print('Confusion matrix, without normalization')

    print(cm)

#     fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 绘制ROC曲线
    ax = fig.add_subplot(122)
    ax.plot(fpr, tpr, label='ROC')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    
    # 整合评估指标
    metrics = pd.DataFrame({"ACC_pred": [acc1], "Specificity": [specificity], "Sensitivity": [sensitivity],
                            "PPV": [ppv], "NPV": [npv], "F1-Score": [f1], "ACC_argmax": [acc2], "AUC": [roc_auc]})
    
    fig.tight_layout()
    return metrics,bestThresh


