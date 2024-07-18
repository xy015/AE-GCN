# 数据处理
import pandas as pd
import numpy as np

# Pyecharts绘图所需函数
import json

from pyecharts import options as opts
from pyecharts.charts import Graph

from pyecharts.globals import CurrentConfig, NotebookType

def countNodes(data, edges):
    nodes = []
    # 计算节点个数
    n = data.shape[0]
    print(n)
    # 通过循环添加节点
    for i in range(n):
        # 获取节点编号
        index = data.index[i]
        # 获取该节点的类别
        label = data['flag'][i]
        # 计算与该节点有引用关系的节点的数量
        value = edges[edges["source"] == index].shape[0]
#         print(i, index, label, value)
        # 由于总的节点数量较大，因此我们根据节点的边的数量进行筛选
        # 若边的数量小于5，则不在我们的关系图中绘制出来
#         if value < 5:
#             continue
        # 节点信息
        node = {
            # 节点名称
            'name': str(index),
            # 将有引用关系的节点的数量作为节点的大小
            'symbolSize': value,
            # 最后图片是否可使用鼠标拖曳
            'draggable': 'True',
            # 节点的数值
            'value': value,
            # 定义节点的类别
            'category': "标签: "+str(label),
            # 在最终的图中不显示节点名称
            'label': {'normal': {'show': False}}
        }
        # 添加该节点的信息
        nodes.append(node)
    return nodes

def countEdges(edges):
    links = []
    # 边的数量
    m = edges.shape[0]
    for j in range(m):
        # 获取一条边两头的两个节点
        source = edges["source"][j]
        target = edges["target"][j]
        # 构造词典存储边的信息
        link = {"source": str(source), "target": str(target)}
        # 添加该条边的信息
        links.append(link)
    return links

def graphVisual(nodes, links, categories):
    graph = (
        # 设置关系图的大小尺寸
        Graph(init_opts=opts.InitOpts(width="1080px", height="720px"))
        .add(
            # 图形名称，一般设置为空字符串即可
            series_name="",
            # 节点信息
            nodes=nodes,
            # 边信息
            links=links,
            # 类别信息
            categories=categories,
            # 调节节点间距离的参数，值越大，节点间的距离会越大
            # 可以尝试替换成[-1, 1000]间的任意数值并查看效果
            repulsion=50,
            # 添加以下参数可以为边设置形状，arrow为箭头，circle为圆点
#             edge_symbol='arrow', # edge_symbol='circle',
            # curve为边的弯曲程度，这里设置为0即直线
            # 一般建议将值设置在0-0.5之间
            # 由于该数据中的边都为双向的，因此将curve设置为大于0的值后，两个节点间会出现两条边，可以自行尝试
            linestyle_opts=opts.LineStyleOpts(curve=0),
        )
        .set_global_opts(
            # 设置图例，竖向排列，在距离左侧2%，距离顶部20%的位置生成图例
            legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
            # 设置图像标题
            title_opts=opts.TitleOpts(title="可视化结果"),
        )
    )
    return graph