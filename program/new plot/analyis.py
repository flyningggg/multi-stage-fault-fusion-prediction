import shapefile
import sys
import os

# 支持从 program 或 new plot 目录运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAM_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)
try:
    from utils.matplotlib_chinese import setup_matplotlib_chinese
    setup_matplotlib_chinese()
except ImportError:
    pass  # 独立运行时可跳过

from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np


# 两点之间的顺时针角度

def angle_calc(x1, y1, x2, y2):
    """ 两点之间的顺时针角度 """
    x = x1 - x2
    y = y1 - y2

    anglez = np.degrees(np.arctan2(x, y))
    if anglez < 0:
        anglez += + 180

    if anglez == 180.0:
        anglez -= 0.01
    return anglez


# LIANGBARSKY THING

def liangbarsky(left, top, right, bottom, x1, y1, x2, y2):
    """ 计算网格中折线的长度
        参数
        -----------
        左、上、右、下	= 框的边界坐标
        x1, y1, x2, y2		= 网格中的点数
        返回
        -----------
        框中折线的长度

        -----------------------------------------------------------------
    """
    dx = x2 - x1
    dy = y2 - y1
    dt0, dt1 = 0, 1

    checks = ((-dx, -(left - x1)), (dx, right - x1), (-dy, -(bottom - y1)), (dy, top - y1))

    for p, q in checks:
        if p == 0:
            p = 0.01
            if q < 0:
                return 0, 0, 0, 0, 0
        dt = q / (p * 1.0)
        if p < 0:
            if dt > dt1:
                return 0, 0, 0, 0, 0
            dt0 = max(dt0, dt)
        else:
            if dt < dt0:
                return 0, 0, 0, 0, 0
            dt1 = min(dt1, dt)

    x2 = x1 + (dt1 * dx)
    y2 = y1 + (dt1 * dy)
    x1 = x1 + (dt0 * dx)
    y1 = y1 + (dt0 * dy)

    distance = np.hypot(x2 - x1, y2 - y1)
    return distance, x1, y1, x2, y2


################################################################################

class FracAnalysisPoly:
    """
        在网格中返回分析的形状文件。
        参数
        -----------
        address : string
            仅包含折线的形状文件的位置
        cell_size : int, float
            网格平方边的大小（公里）
        angle_divs : int
            180 度角分位数
       属性
        -----------
        X, Y	: numpy.ndarray
            具有非零点的网格的 x 和 y 坐标数组
        N : numpy.ndarray
            平方计数数组，除以角度箱
        N_total : numpy.ndarray
            每平方真实裂缝数数组（不包括>1角度箱）中的裂缝重复计数）
        L : numpy.ndarray
            裂缝长度的数组（以平方为单位），以公里为单位，除以角度箱
        Number_Anisotropy : numpy.ndarray
            数字各向异性数组（每平方 1 个值）
        功能
        -----------
        save_output:
            将属性保存到形状文件
        -----------------------------------------------------------------
    """

    def __init__(self, address, cell_size, angle_divs):
        self.address = address
        self.cell_size = cell_size
        self.angle_divs = angle_divs

        # 读数据
        sf = shapefile.Reader(address)
        shapes = sf.shapes()

        x_coord, y_coord = [], []
        for i in shapes[::]:
            a = np.array(i.points)
            x_coord.append(a[:, 0])
            y_coord.append(a[:, 1])

        if type(angle_divs) != int:
            print("\n The argument 'angle_divs' needs to be int\n")
            return

        km = 1000

        # 获取绘图的边界
        maxx = max([item for sublist in x_coord for item in sublist])
        minx = min([item for sublist in x_coord for item in sublist])
        maxy = max([item for sublist in y_coord for item in sublist])
        miny = min([item for sublist in y_coord for item in sublist])

        x1 = 0
        x2 = (int((maxx / km) / cell_size) * float(cell_size)) + cell_size
        y1 = 0
        y2 = (int((maxy / km) / cell_size) * float(cell_size)) + cell_size

        x_vals = (np.arange(x1, x2 + cell_size, cell_size) * km)
        y_vals = (np.arange(y1, y2 + cell_size, cell_size) * km)

        meshedx, meshedy = np.meshgrid(x_vals, y_vals)

        meshedxx = np.reshape(meshedx, (len(x_vals) * len(y_vals), 1), order="C")
        meshedyy = np.reshape(meshedy, (len(x_vals) * len(y_vals), 1), order="C")

        a = np.zeros((len(x_vals) * len(y_vals), 4))  # a 将包含正方形的 X 和 Y 坐标
        b = np.zeros((len(x_vals) * len(y_vals), angle_divs))  # b 将是按角度箱划分的 2D 长度列表
        c = np.zeros((len(x_vals) * len(y_vals), angle_divs))  # c 将是按角度箱列出的数字的 2D 列表
        d = np.zeros(len(a))  # d将是真实的数字密度（以防止重复计算）

        a[:, 0] = meshedxx[:, 0]
        a[:, 1] = meshedyy[:, 0]
        temp_num = []  # temp_num将填充以下形式的元组（方形索引、折线 ID）

        for x, y, polyline_id in zip(x_coord, y_coord, range(len(x_coord))):
            temp_dens = []  # temp_dens用于获取每angle_bin的密度

            for i in np.arange(0, len(x) - 1, 1):  # 对于每条折线：
                xi = int((x[i] / km) / cell_size) * float(cell_size)  # 起始方块 X
                xi_1 = int((x[i + 1] / km) / cell_size) * float(cell_size)  # 结束平方 x
                yi = int((y[i] / km) / cell_size) * float(cell_size)  # 起始方格 Y
                yi_1 = int((y[i + 1] / km) / cell_size) * float(cell_size)  # 结束平方 Y

                if (xi == xi_1) and (yi == yi_1):  # 线停留在单个正方形

                    x_index = ((xi - x1) / cell_size)
                    y_index = ((yi - y1) / cell_size)
                    index = int((x_index + (y_index * len(x_vals))))  # 平方指数

                    distance = np.hypot(x[i] - x[i + 1], y[i] - y[i + 1])  # 距离

                    frac_angle = angle_calc(x[i], y[i], x[i + 1], y[i + 1])  # 角度

                    # 在正确的方格中加到 B 中，angle_index：
                    angle_bin = int(int(frac_angle) / int(180 / angle_divs))
                    b[index][angle_bin] += distance

                    a[index][3] += distance  # 加入distance
                    a[index][2] += 1  # 添加到第 3 行（用于删除空方块

                    # 将方形索引和角度箱添加到temp_dens：
                    temp_dens.append((index, int(int(frac_angle) / int(180 / angle_divs))))
                    # 将平方索引和polyline_id添加到temp_num：
                    temp_num.append((index, polyline_id))

                else:  # 如果折线从一个正方形开始，到另一个正方形结束：
                    # 在线可以居住的每个正方形上循环
                    for l in np.arange(min(xi, xi_1), max(xi, xi_1) + cell_size, cell_size):
                        for j in np.arange(min(yi, yi_1), max(yi, yi_1) + cell_size, cell_size):

                            x_index = (l - x1) / cell_size
                            y_index = (j - y1) / cell_size

                            index = int((x_index + (y_index * len(x_vals))))

                            # 使用liangbarsky计算出每个方块的距离
                            distance, xx1, yy1, xx2, yy2 = liangbarsky(l * 1000, (j + cell_size) * 1000,
                                                                       (l + cell_size) * 1000, j * 1000, x[i], y[i],
                                                                       x[i + 1], y[i + 1])
                            if distance != 0:
                                # if distance = 0, 保存相关数据
                                frac_angle = angle_calc(xx1, yy1, xx2, yy2)

                                angle_bin = int(int(frac_angle) / int(180 / angle_divs))
                                b[index][angle_bin] += distance

                                temp_dens.append((index, int(int(frac_angle) / int(180 / angle_divs))))
                                temp_num.append((index, polyline_id))

                            a[index][3] += distance
                            a[index][2] += 1

            # 遍历temp_dens，将数据提取到C ->按角度箱编号
            for h in set(temp_dens):
                c[h[0]][h[1]] += 1

        true_list = []  # 布尔值列表以摆脱空方块
        for row in a:
            if row[2] == 0 or row[3] == 0:
                true_list.append(False)
            else:
                true_list.append(True)

        true_list = np.array(true_list)

        temp_num = list(set(temp_num))
        for number in temp_num:
            d[number[0]] += 1

        a = a[true_list]
        b = b[true_list]
        c = c[true_list]
        d = d[true_list]

        X, Y = a[:, 0], a[:, 1]

        self.X = X
        self.Y = Y
        self.N = c
        self.L = b / 1000.
        self.N_total = d
        self.__name__ = "FracAnalysisPoly"
        self.GDF = np.sum(self.N, axis=0) / np.sum(self.N)
        self.Number_Anisotropy = np.max(self.N, axis=1) / np.array([min([x for x in i if x > 0]) for i in self.N])

    def save_output(self, address_out):
        """
            创建属性的形状文件
            参数
            -----------
            address_out：字符串
                要写出的形状文件的位置
            -----------------------------------------------------------------
        """
        # 设置形状文件编写器并创建空字段
        w = shapefile.Writer(str(shapefile.POINT))
        w.autoBalance = 1  # 确保属性匹配
        w.field('X', 'F', 10, 8)
        w.field('Y', 'F', 10, 8)

        for i in range(len(self.N.T)):
            header_string = "N" + str(i)
            w.field(header_string, "F", 10, 8)

        for i in range(len(self.N.T)):
            header_string = "L" + str(i)
            w.field(header_string, "F", 10, 8)

        w.field('N_tot', 'F', 10, 8)
        w.field("N_Anisotropy", "F", 10, 8)

        for index, value in enumerate(self.X):
            w.point(self.X[index], self.Y[index])

            zzz = [self.X[index], self.Y[index]]
            for j in self.N[index]:
                zzz.append(j)
            for j in self.L[index]:
                zzz.append(j / 1000.)

            zzz.append(self.N_total[index])
            zzz.append(self.Number_Anisotropy[index])

            w.record(*zzz)

        # 保存shapefile
        # w.save(address_out)
        # print("Saved output to " + address_out)


################################################################################

class FracAnalysisPoint:
    """
        在网格中返回分析的形状文件。
        参数
        -----------
        address : string
            仅包含点的形状文件的位置
        cell_size : int, float
            网格平方边的大小（公里）
        angle_divs : int
            180 度角分位数
        属性
        -----------
        X, Y	: numpy.ndarray
            具有非零点的网格的 x 和 y 坐标数组
        N : numpy.ndarray
            平方计数数组，除以角度箱
        N_total : numpy.ndarray
            每平方真实裂缝数数组（不包括>1角度箱）中的裂缝重复计数）
        Number_Anisotropy : numpy.ndarray
            数字各向异性数组（每平方 1 个值）
        功能
        -----------
        save_output:
            将属性保存到形状文件
        -----------------------------------------------------------------
    """

    def __init__(self, address, cell_size, angle_divs):

        self.address = address
        self.cell_size = cell_size
        self.angle_divs = angle_divs

        sf = shapefile.Reader(address)

        data = list(sf.records())
        x_input = [x[0] for x in data]
        y_input = [y[1] for y in data]
        strike_input = [z[2] for z in data]

        if type(angle_divs) != int:
            print("\n The argument 'angle_divs' needs to be int\n")
            return

        km = 1000.

        # 获取绘图的边界
        maxx = max(x_input)
        minx = min(x_input)
        maxy = max(y_input)
        miny = min(y_input)

        x1 = 0
        x2 = (int((maxx / km) / cell_size) * float(cell_size)) + cell_size
        y1 = 0
        y2 = (int((maxy / km) / cell_size) * float(cell_size)) + cell_size

        x_vals = (np.arange(x1, x2 + cell_size, cell_size) * km)
        y_vals = (np.arange(y1, y2 + cell_size, cell_size) * km)

        meshedx, meshedy = np.meshgrid(x_vals, y_vals)

        meshedxx = np.reshape(meshedx, (len(x_vals) * len(y_vals), 1), order="C")
        meshedyy = np.reshape(meshedy, (len(x_vals) * len(y_vals), 1), order="C")

        a = np.zeros((len(x_vals) * len(y_vals), 3))  # a 将包含正方形的 X 和 Y 坐标
        c = np.zeros((len(x_vals) * len(y_vals), angle_divs))  # c 将是按角度箱列出的数字的 2D 列表

        a[:, 0] = meshedxx[:, 0]
        a[:, 1] = meshedyy[:, 0]

        for x, y, strike in zip(x_input, y_input, strike_input):
            temp_dens = []  # temp_dens用于获取每angle_bin的密度

            xi = int((x / km) / cell_size) * float(cell_size)  # 起始方块 X
            yi = int((y / km) / cell_size) * float(cell_size)  # 起始方块 y

            x_index = ((xi - x1) / cell_size)
            y_index = ((yi - y1) / cell_size)
            index = int((x_index + (y_index * len(x_vals))))  # 平方指数

            a[index][2] += 1  # 添加到第 3 行（用于删除空方块
            try:
                c[index][int(strike) / int(180 / angle_divs)] += 1
            except:
                c[index][(int(strike) - 1) / int(180 / angle_divs)] += 1

        true_list = []  # 目录
        for row in a:
            if row[2] == 0:
                true_list.append(False)
            else:
                true_list.append(True)

        true_list = np.array(true_list)

        a = a[true_list]
        c = c[true_list]
        X, Y = a[:, 0], a[:, 1]

        self.A = a
        self.X = X
        self.Y = Y
        self.N = c
        self.GDF = np.sum(self.N, axis=0) / np.sum(self.N)
        self.N_total = np.max(self.N, axis=1)
        self.__name__ = "FracAnalysisPoint"
        self.Number_Anisotropy = np.max(self.N, axis=1) / np.array([min([x for x in i if x > 0]) for i in self.N])

    def save_output(self, address_out):

        """
           创建属性的形状文件

            参数
            -----------
            address_out : string
                仅包含折线的形状文件的位置
            -----------------------------------------------------------------
        """
        # 设置形状文件编写器并创建空字段
        w = shapefile.Writer(shapefile.POINT)
        w.autoBalance = 1  # 确保gemoetry和属性匹配
        w.field('X', 'F', 10, 8)
        w.field('Y', 'F', 10, 8)

        for i in range(len(self.N.T)):
            header_string = "N" + str(i)
            w.field(header_string, "F", 10, 8)

        w.field('N_tot', 'F', 10, 8)
        w.field("N_Anisotropy", "F", 10, 8)

        for index, value in enumerate(self.X):
            w.point(self.X[index], self.Y[index])

            zzz = [self.X[index], self.Y[index]]
            for j in self.N[index]:
                zzz.append(j)

            zzz.append(self.N_total[index])
            zzz.append(self.Number_Anisotropy[index])

            w.record(*zzz)

        # 保存 shapefile
        w.save(address_out)
        print("Saved output to " + address_out)


################################################################################

def FancyPlot(FracAnalyzed, Rose=True, Fractures=True, Patches=False, Circles=False, SquareNumbers=False, Title="",
              FigureNumber=1):
    """绘制point_number_density
        参数
        -----------
        FracAnalyzed : FracAnalysisPoly/Point 对象的列表或单个实例可以是不同 FracAnalysisPoint 和/或 FracAnalysisPoly 的列表
 对象。这些对象必须具有相同的cell_size和anlge_bins
        Rose : Boolean
            默认为 True
            包括每个正方形的玫瑰图
        Fractures : Boolean
           默认为 True
           包括图中的“原始”折线和点
        Patches : "Number", "Length", "NumberAnisotropy" or False
            默认为False
            包括表示长度密度、数字密度或带颜色条的数字各向异性的正方形
        Circles : Boolean
            默认为False
            包括每个玫瑰图周围的圆圈，按角度箱划分为扇区。圆圈表示 25、50、75 和 100% 的比例。
        SquareNumbers : Boolean
            默认为False
            包括每个方块中的数字。
        Title: string
            默认为空
            图的标题
        Returns
        --------
        一幅图
        -----------------------------------------------------------------
    """
    if type(FracAnalyzed) != list:
        FracAnalyzed = [FracAnalyzed]

    cell_size = FracAnalyzed[0].cell_size

    fig = plt.figure(FigureNumber)
    fig.clf()
    ax = fig.add_subplot(111)

    minx = min([min(temp.X) for temp in FracAnalyzed])
    maxx = max([max(temp.X) for temp in FracAnalyzed])

    miny = min([min(temp.Y) for temp in FracAnalyzed])
    maxy = max([max(temp.Y) for temp in FracAnalyzed])

    plt.xlim([minx, maxx + (cell_size * 1000)])
    plt.ylim([miny, maxy + (cell_size * 1000)])
    plt.grid(True)

    ############################################################################
    # 画 Fractures
    if Fractures == True:
        for classs in FracAnalyzed:

            if classs.__name__ == "FracAnalysisPoly":
                sf = shapefile.Reader(classs.address)
                shapes = sf.shapes()

                for i in shapes[::]:
                    a = np.array(i.points)
                    plt.plot(a[:, 0], a[:, 1], zorder=2)

            if classs.__name__ == "FracAnalysisPoint":
                sf = shapefile.Reader(classs.address)

                data = list(sf.records())
                x_input = [x[0] for x in data]
                y_input = [y[1] for y in data]

                plt.scatter(x_input, y_input, zorder=2, color="white", edgecolor="black")

    ############################################################################
    # 画Patches Squares
    cmapp = plt.cm.jet

    # Number Density
    if Patches == "Number":
        max_value = max([np.max(classs.N_total) for classs in FracAnalyzed])

        for classs in FracAnalyzed:
            X = classs.X
            Y = classs.Y
            ZZ = classs.N_total

            s = plt.scatter(X + (500 * cell_size), Y + (500 * cell_size), c=ZZ, cmap=cmapp, s=1)

            for x, y, c, number in zip(X, Y, ZZ, range(len(ZZ))):
                ax.add_artist(plt.Rectangle(xy=(x, y), color=cmapp(c / max_value), width=cell_size * 1000,
                                            height=cell_size * 1000, alpha=0.8))

        cbar = plt.colorbar(s)
        cbar.set_label("每{}平方公里的裂缝数".format(cell_size), fontsize=16, rotation=90)
        plt.clim(0, max_value)


    # --#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#
    # Length Density
    elif Patches == "Length":
        max_value = max([np.max(np.sum(classs.L, axis=1)) for classs in FracAnalyzed])

        for classs in FracAnalyzed:
            X = classs.X
            Y = classs.Y
            ZZ = np.sum(classs.L, axis=1)

            s = plt.scatter(X + (500 * cell_size), Y + (500 * cell_size), c=ZZ, cmap=cmapp, s=1)

            for x, y, c, number in zip(X, Y, ZZ, range(len(ZZ))):
                ax.add_artist(plt.Rectangle(xy=(x, y), color=cmapp(c / max_value), width=cell_size * 1000,
                                            height=cell_size * 1000, alpha=0.8))

        cbar = plt.colorbar(s)
        cbar.set_label("Length of Fractures (km) per\n   {} km squared".format(cell_size), fontsize=16, rotation=90)
        plt.clim(0, max_value)


    # --#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#
    # Number Anisotropy
    elif Patches == "NumberAnisotropy":
        max_value = max([np.max(classs.Number_Anisotropy) for classs in FracAnalyzed])

        for classs in FracAnalyzed:
            X = classs.X
            Y = classs.Y
            ZZ = classs.Number_Anisotropy

            s = plt.scatter(X + (500 * cell_size), Y + (500 * cell_size), c=ZZ, cmap=cmapp, s=0.1)

            for x, y, c, number in zip(X, Y, ZZ, range(len(ZZ))):
                ax.add_artist(plt.Rectangle(xy=(x, y), color=cmapp(c / max_value), width=cell_size * 1000,
                                            height=cell_size * 1000, alpha=0.8))

        cbar = plt.colorbar(s)
        cbar.set_label("Number Anisotropy per\n   {} km squared".format(cell_size), fontsize=16, rotation=90)
        plt.clim(0, max_value)

    ############################################################################
    # Add numbers to the squares
    if SquareNumbers == True:
        for classs in FracAnalyzed:
            X = classs.X
            Y = classs.Y

            for x, y, number in zip(X, Y, range(len(X))):
                plt.text(x, y + ((0.7 * cell_size) * 1000), str(number), color="white", zorder=100000, fontsize=10)

    ############################################################################
    # 画玫瑰图
    if Rose == True:
        for classs in FracAnalyzed:

            X = classs.X
            Y = classs.Y
            Z = classs.N

            patches = []
            for x, y, l in zip(X, Y, Z):
                L = np.sqrt(l / sum(l)) * cell_size * 500
                increment = 180. / len(L)
                start = 0

                startx, starty = (x + (0.5 * cell_size) * 1000), (y + (0.5 * cell_size) * 1000)
                for i in L:
                    end = start + increment
                    wedgez = Wedge((startx, starty), i, 90 - end, 90 - start)
                    patches.append(wedgez)

                    wedgez = Wedge((startx, starty), i, (270 - end), (270 - start))
                    patches.append(wedgez)
                    start = start + increment

                if Circles == True:
                    for classs in FracAnalyzed:
                        radius = cell_size * 0.5
                        # 玫瑰图周围的圆圈
                        for i in (.25, .5, .75, 1):
                            circle1 = plt.Circle((startx, starty), np.sqrt(i) * 1000.0 * radius, color="black",
                                                 fill=False, lw=1.5, alpha=0.25)
                            fig.gca().add_artist(circle1)

                        my_angle = 0.

                        # Spokes in these circles
                        for i in range(len(L) * 2):
                            plt.plot([startx, startx + (1000 * radius * np.sin(np.deg2rad(90 - 180 - 90 - my_angle)))],
                                     [starty, starty + (1000. * radius * np.sin(np.deg2rad(90 - my_angle)))], color="k",
                                     lw=1.5, alpha=0.25)
                            my_angle += increment

            p = PatchCollection(patches, alpha=0.5, color="black", zorder=1000)
            ax.add_collection(p)
    ax.set_aspect('equal')
    plt.savefig('myplot10.png')
    plt.show()


################################################################################


def FancyPlotTotals(FracAnalyzed, Fractures=True, Circles=True, FigureNumber=1):
    """Plots the point_number_density
        参数
        -----------
        FracAnalyzed : FracAnalysisPoly/Point 对象的列表或单个实例可以是不同 FracAnalysisPoint 和/或 FracAnalysisPoly 对象的列表。
        这些对象必须具有相同的cell_size和anlge_bins
        Fractures : Boolean
            默认为 True
            包括图中的“原始”折线和点
        Circles : Boolean
            默认为 False
            包括每个玫瑰图周围的圆圈，按角度箱划分为扇区。圆圈表示 25、50、75 和 100% 的比例。
        Title : string
            默认为空
            Title of the plot
        FigureNumber : int
            创建的数字数量
        Returns
        --------
        另一幅图
        -----------------------------------------------------------------
    """

    if type(FracAnalyzed) != list:
        FracAnalyzed = [FracAnalyzed]

    fig = plt.figure(FigureNumber)
    fig.clf()
    ax = fig.add_subplot(111)

    minx = min([min(temp.X) for temp in FracAnalyzed])
    maxx = max([max(temp.X) for temp in FracAnalyzed])

    miny = min([min(temp.Y) for temp in FracAnalyzed])
    maxy = max([max(temp.Y) for temp in FracAnalyzed])

    cell_size = FracAnalyzed[0].cell_size
    plt.xlim([minx, maxx + (cell_size * 1000)])
    plt.ylim([miny, maxy + (cell_size * 1000)])
    plt.grid(True)

    ############################################################################
    # 拉伸断裂
    if Fractures == True:
        for classs in FracAnalyzed:

            if classs.__name__ == "FracAnalysisPoly":
                sf = shapefile.Reader(classs.address)
                shapes = sf.shapes()

                for i in shapes[::]:
                    a = np.array(i.points)
                    plt.plot(a[:, 0], a[:, 1], zorder=100)

            if classs.__name__ == "FracAnalysisPoint":
                sf = shapefile.Reader(classs.address)

                data = list(sf.records())
                x_input = [x[0] for x in data]
                y_input = [y[1] for y in data]

                plt.scatter(x_input, y_input, zorder=100)

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # 绘制玫瑰图
    for classs in FracAnalyzed:

        pointsx, pointsy = [], []
        sf = shapefile.Reader(classs.address)
        shapes = sf.shapes()

        for i in shapes[::]:
            a = np.array(i.points)
            pointsx.extend(a[:, 0])
            pointsy.extend(a[:, 1])

        X = classs.X
        X_max, X_min = max(X), min(X)
        X_width = X_max - X_min

        Y = classs.Y
        Y_max, Y_min = max(Y), min(Y)
        Y_width = Y_max - Y_min

        cell_size = min([X_width, Y_width]) / 1000.0
        startx, starty = np.mean(pointsx), np.mean(pointsy)

        Z = np.sum(classs.N, axis=0)

        radius = cell_size / 2.
        a = Z / np.sum(Z)
        L = np.sqrt(a) * radius * 1000.
        increment = 180. / len(L)
        start = 0
        patches = []

        for i in L:
            end = start + increment
            wedgez = Wedge((startx, starty), i, 90 - end, 90 - start)
            patches.append(wedgez)

            wedgez = Wedge((startx, starty), i, (270 - end), (270 - start))
            patches.append(wedgez)
            start = start + increment

        p = PatchCollection(patches, alpha=0.5, color="black", zorder=1000)
        ax.add_collection(p)

        # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
        # 添加圈子
        if Circles == True:
            # 玫瑰图周围的圆圈
            for i in (.25, .5, .75, 1):
                circle1 = plt.Circle((startx, starty), np.sqrt(i) * 1000.0 * radius, color="black", fill=False, lw=1.5,
                                     alpha=0.5)
                fig.gca().add_artist(circle1)

            my_angle = 0.

            # 这些圈子中的辐条
            for i in range(len(L) * 2):
                plt.plot([startx, startx + (1000 * radius * np.sin(np.deg2rad(90 - 180 - 90 - my_angle)))],
                         [starty, starty + (1000. * radius * np.sin(np.deg2rad(90 - my_angle)))], color="k", lw=1.5,
                         alpha=0.5)
                my_angle += increment

################################################################################


# FancyPlot(b, Rose = True, Fractures = True, Patches = "NumberAnisotropy", Circles = True, SquareNumbers = True, FigureNumber = 1)
# FancyPlot(b)
# cell_size, angle_bins = 20, 3
# address1 = "venv/Data/g_10.shp"
# a = FracAnalysisPoly(address1, cell_size, angle_bins)
# analysed_list = [a, b]
# FancyPlot(analysed_list, Rose = True, Fractures = True, Patches = "NumberAnisotropy", Circles = True, SquareNumbers = True, FigureNumber = 1)
# address3 = "venv/Data/Area_3_polygons.dbf"
# p = FracAnalysisPoint(address3, cell_size ,angle_bins)
# analysed_list.append(p)
# FancyPlot(analysed_list, Rose = True, Fractures = True, Patches = "Number", Circles = False, SquareNumbers = False, FigureNumber = 1)
# FancyPlotTotals(analysed_list, Fractures = True, Circles = True, FigureNumber = 1)
# address_out = "/my_output"
# b.save_output(address_out)
# address = "dataset/json2shp3.shp"
# cell_size, angle_bins = 40, 5
# b = FracAnalysisPoly(address, cell_size, angle_bins)
# FancyPlot(b, Patches="Number")
