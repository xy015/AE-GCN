# coding=utf-8
import os
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf


# 前向传递函数，输入数据，返回模型结果
@tf_utils.function
def forward(model, graph, training=False):
    return model(graph, training)


# 计算损失函数
@tf_utils.function
def compute_loss(graph, logits, vars):
    # 使用 sigmoid 交叉熵损失函数计算损失
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.convert_to_tensor(graph.y, dtype=tf.float32)
    )
    # 提取所有卷积核变量（weights）并计算 L2 正则化项
    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 1e-5


# 设置训练函数
@tf_utils.function
def train_step(model, graph, optimizer):
    # 计算损失函数
    with tf.GradientTape() as tape:
        logits = forward(model, graph, training=True)
        loss = compute_loss(graph, logits, tape.watched_variables())
    # 计算梯度
    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    #梯度下降，更新模型参数
    optimizer.apply_gradients(zip(grads, vars))
    return loss


# 设置模型评估函数
@tf_utils.function
def evaluate(model, graph):
    # 预测全部数据的结果
    logits = forward(model, graph) ##神经网络推理，forward与感知机相比，多了一个激活函数的模块
    # 获取预测标签
    y_pred = tf.argmax(logits, axis=-1, output_type=tf.int64)
    # 获取真实标签
    y_true = tf.argmax(graph.y, axis=-1, output_type=tf.int64)
    
    # 计算准确率
    corrects = tf.equal(y_pred, y_true) ##返回(x == y)元素的真值
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32)) ##cast为bool型转换为int或float
    ##reduce_mean为张量tensor沿着指定的数轴（tensor的某一维度）上的平均值，主要用作降维或者计算tensor（图像）的平均值。
    return accuracy


# 设置模型评估函数
@tf_utils.function
def getPreds(model, graph, index=None):
    # 预测全部数据的结果
    logits = forward(model, graph)
    # 获取预测标签
    y_pred = tf.argmax(logits, axis=-1, output_type=tf.int64)
    # 获取真实标签
    y_true = tf.argmax(graph.y, axis=-1, output_type=tf.int64)
    # 返回真实标签、预测标签、预测结果
    return y_true, y_pred, logits


