# coding=utf-8
import os
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf



# 前向传递函数，输入数据，返回模型结果
@tf_utils.function
def forward(model, graph, training=False):
    return model([graph.x, graph.edge_index], training=training)


# 计算损失函数
@tf_utils.function
def compute_loss(graph, num_classes, logits, mask_index, vars):
    # 获取特定编号样本的预测结果
    masked_logits = tf.gather(logits, mask_index)
    # 获取特定编号样本的真实标签
    masked_labels = tf.gather(graph.y, mask_index)
    # 计算损失函数
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


# 设置训练函数
@tf_utils.function
def train_step(model, graph, num_classes, optimizer, train_index):
    # 计算损失函数
    with tf.GradientTape() as tape:
        logits = forward(model, graph, training=True)
        loss = compute_loss(graph, num_classes, logits, train_index, tape.watched_variables())
    # 计算梯度
    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    # 梯度下降
    optimizer.apply_gradients(zip(grads, vars))
    return loss


# 设置模型评估函数
@tf_utils.function
def evaluate(model, graph, index):
    # 预测全部数据的结果
    logits = forward(model, graph)
    # 获取测试集样本的预测结果
    masked_logits = tf.gather(logits, index)
    # 获取测试集样本的标签
    masked_labels = tf.gather(graph.y, index)
    
    # 获取预测标签
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
    
    # 计算准确率
    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


# 设置模型评估函数
@tf_utils.function
def getPreds(model, graph, index):
    # 预测全部数据的结果
    logits = forward(model, graph)
    # 获取测试集样本的预测结果
    masked_logits = tf.gather(logits, index)  # [y1,y2]
    # 获取测试集样本的标签
    masked_labels = tf.gather(graph.y, index)
    # 获取预测标签
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int64)
    # 返回真实标签、预测标签、预测结果
    return masked_labels, y_pred, masked_logits


