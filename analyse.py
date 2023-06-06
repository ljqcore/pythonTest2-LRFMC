import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import wx
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


# 数据预处理
# 1. 数据清洗，过滤不符合规则的数据
def filterData():
    data = pd.read_csv("../LRFMC/air_data.csv", encoding="gb18030")  # 导入航空数据
    print("原始数据的形状为：", data.shape)
    print(data)
    # 要求1、丢弃票价为空的记录。 SUM_YR_1是客户在第一年乘坐该航空公司的总票价支出
    data = data[data["SUM_YR_1"].notnull() & data["SUM_YR_2"].notnull()]  # 票价非空值才保留
    print("第一次数据过滤后：", data.shape)
    # 要求2、丢弃票价为0、平均折扣率不为0、总飞行千米数大于0的记录: 把票价不为0，飞行公里为0，平均折扣为0的找到并丢弃。
    doc1 = (data["SUM_YR_1"] != 0) | (data["SUM_YR_2"] != 0)
    doc2 = (data["SEG_KM_SUM"] == 0) & (data["avg_discount"] == 0)  # 飞行公里==0,平均折扣==0的
    data = data[doc1 | doc2]
    print("第二次数据过滤后：", data.shape)
    return data


# 2. 数据变换：提取LRFMC五个指标
def getMetrics(air_data):
    '''
    L=LOAD_TIME-FFP_DATE
    会员入会时间距观测窗口结束的月数=观测窗口的结束时间-入会时间 [月]
    R=LAST_TO_END
    客户最近一次乘坐公司飞机距观测窗口结束的月数=最后一次乘机时间至观察窗口末端时长 [月]
    F=FLIGHT_COUNT
    客户在观测时间内乘坐公司飞机的次数=观测窗口的飞行次数 [次]
    M=SEG_KM_SUM
    客户在观测时间内在公司累计的飞行里程=观测窗口的总飞行里数 [公里]
    C=AVG_DISCOUNT
    客户在观测时间内乘坐舱位所对应的折扣系数的平均值=平均折扣率 [无]
    '''
    # L=LOAD_TIME-FFP_DATE
    air_data["LOAD_TIME"] = pd.to_datetime(air_data["LOAD_TIME"])  # 要按pd.datetime格式转换时间格式
    air_data["FFP_DATE"] = pd.to_datetime(air_data["FFP_DATE"])
    air_data["会员入会时间L"] = air_data["LOAD_TIME"] - air_data["FFP_DATE"]
    print(air_data["会员入会时间L"])
    # 单位转化为月
    mon = []
    for i in air_data["会员入会时间L"]:
        months = int(i.days / 30)
        mon.append(months)
    air_data["L"] = np.array(mon)
    print(air_data["L"])

    #  R=LAST_TO_END
    air_data["R"] = int(air_data["LAST_TO_END"] / 30)
    print(air_data["R"])

    #  F=FLIGHT_COUNT; M=SEG_KM_SUM; C=AVG_DISCOUNT
    my_data = air_data[["L", "R", "FLIGHT_COUNT", "SEG_KM_SUM", "avg_discount"]]
    my_data = my_data.rename(columns={"L": "L", "R": "R", "FLIGHT_COUNT": "F", "SEG_KM_SUM": "M", "avg_discount": "C"})
    print(my_data)
    # 标准化处理：处理后，所有特征的均值为0，标准差为1
    my_data = StandardScaler().fit_transform(my_data)
    np.savez("airline_scale.npz", my_data)
    print(my_data)
    return my_data


# 读入客户数据
def createK2():
    airline_scale = np.load('../LRFMC/airline_scale.npz')['arr_0']
    print(type(airline_scale))  # ndarray类型

    k = 5  # 聚类中心数
    kmeans_model = KMeans(n_clusters=k, random_state=123)
    # 聚类中心数，并行的CPU核的数量，随机数种子
    fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练
    print("聚类中心：", kmeans_model.cluster_centers_)  # 查看聚类中心
    print("聚类标签：", kmeans_model.labels_)  # 查看样本的类别标签
    # 统计不同类别样本的数目
    r1 = pd.Series(kmeans_model.labels_).value_counts()
    print("最终每个类别的数目：\n", r1)
    # 测试这一组数据会被分配在哪个类别中
    result = kmeans_model.predict([[1.5, 1.5, 1.5, 1.5, 1.5]])
    print(result)

    # 对建立的Kmeans模型进行评价
    # 聚类模型评价:FMI
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=123).fit(airline_scale)
        score = fowlkes_mallows_score(kmeans_model.labels_, kmeans.labels_)
        print("航空公司数据聚%d类FMI评价分值为：%f" % (i, score))

    # 聚类模型评价：silhouette
    silhouseScore = []
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=123).fit(airline_scale)
        score = silhouette_score(airline_scale, kmeans.labels_)
        print("航空公司数据聚%d类silhouette评价分值：%f" % (i, score))
        silhouseScore.append(score)

    # 聚类模型评价：calinski_harabasz
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=123).fit(airline_scale)
        score = calinski_harabasz_score(airline_scale, kmeans.labels_)
        print("航空公司数据聚%d类calinski_harabasz评价分值：%f" % (i, score))

    # 绘制雷达图
    r = pd.Series(['客户群1', '客户群2', '客户群3', '客户群4', '客户群5'])
    labels = np.array(["L", "R", "F", "M", "C"])
    fig = plt.figure()
    datas = kmeans_model.cluster_centers_
    ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 加上第一个角度值，使其闭合
    labels = np.concatenate((labels, [labels[0]]))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(len(datas)):
        data = np.concatenate((datas[i], [datas[i][0]]))  # 将最后一个数据点作为起点，使其闭合
        ax.plot(angles, data, linewidth=2, label=r[i])
    ax.legend(loc=1)
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    ax.set_title("客户群特征分析图", va='bottom', fontproperties="SimHei")
    ax.grid(True)
    plt.show()


class MyFrame(wx.Frame):

    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(670, 630))

        # 创建主面板
        self.panel = wx.Panel(self)

        # 创建四个按钮
        self.btn_fmi = wx.Button(self.panel, label='FMI评价', pos=(40, 15))
        self.btn_silhouette = wx.Button(self.panel, label='silhouette评价', pos=(160, 15))
        self.btn_calinski_hara = wx.Button(self.panel, label='calinski_hara评价', pos=(320, 15))
        self.btn_customer_seg = wx.Button(self.panel, label='客户群分析', pos=(500, 15))

        # 绑定事件处理函数
        self.btn_fmi.Bind(wx.EVT_BUTTON, self.on_btn_fmi)
        self.btn_silhouette.Bind(wx.EVT_BUTTON, self.on_btn_silhouette)
        self.btn_calinski_hara.Bind(wx.EVT_BUTTON, self.on_btn_calinski_hara)
        self.btn_customer_seg.Bind(wx.EVT_BUTTON, self.on_btn_customer_seg)

        # 创建静态文本和图片显示控件
        self.static_text = wx.StaticText(self.panel, label='图片显示区域', pos=(10, 50))
        self.image_ctrl = wx.StaticBitmap(parent=self.panel, pos=(10, 80))

    def on_btn_fmi(self, event):
        image_path = 'FMI评价指标分析图.jpg'
        bmp = wx.Bitmap(image_path)
        self.image_ctrl.SetBitmap(bmp)

    def on_btn_silhouette(self, event):
        image_path = 'silhouette评价指标分析图.jpg'
        bmp = wx.Bitmap(image_path)
        self.image_ctrl.SetBitmap(bmp)

    def on_btn_calinski_hara(self, event):
        image_path = 'calinski_harabasz评价指标分析图.jpg'
        bmp = wx.Bitmap(image_path)
        self.image_ctrl.SetBitmap(bmp)

    def on_btn_customer_seg(self, event):
        image_path = '客户群特征分析图.jpg'
        bmp = wx.Bitmap(image_path)
        self.image_ctrl.SetBitmap(bmp)


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame(parent=None, title='分析图展示')
    frame.Show()
    app.MainLoop()
