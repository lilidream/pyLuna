def mapper(x, x1, x2, X1, X2):
    '''
    一维线性映射函数，返回在x1-x2中的x，映射到X1-X2后的值。
    :return: float
    '''
    return ((X2 - X1) * (x - x1) / (x2 - x1)) + X1


def mapper_array(array, x1, x2, X1, X2):
    '''
    数组映射，将数组数值从[x1,x2]映射至[X1,X2]
    :return: npArray
    '''
    import numpy as np
    array = np.array(array)
    def m(x, x1, x2, X1, X2): return ((X2 - X1) * (x - x1) / (x2 - x1)) + X1
    vfunc = np.vectorize(m)
    return vfunc(array, x1, x2, X1, X2)


def spacial_freq(x, y, minX, maxX, minY, maxY, xSpace, ySpace, value=None, grid=False):
    '''
    计算散点在空间格点上的频率分布(格点左闭右开)
    :param x: list, 散点的x坐标
    :param y: list, 散点的y坐标
    :param minX: 空间范围的x的最小值
    :param maxX: 空间范围的x的最大值
    :param minY: 空间范围的y的最小值
    :param maxY: 空间范围的y的最大值
    :param xSpace: 空间范围的统计格点的x宽度的间隔
    :param ySpace: 空间范围的统计个点的y宽度的间隔
    :param grid: 返回对应格点的坐标，return 为 result,x,y
    :param value: list, 统计频率散点的加权值
    :return: numpy.array, 空间格点的频数矩阵，从(minX,maxY)至(maxX,minY)(认知排序)
    '''
    import numpy as np
    # 创建格点场
    xLength = int((maxX - minX) / xSpace)
    yLength = int((maxY - minY) / ySpace)
    spcaialFreq = np.zeros((yLength, xLength))
    lon = np.arange(minX, maxX, xSpace) + xSpace / 2
    lat = np.arange(maxY, minY, -ySpace) - ySpace / 2

    for i in range(len(x)):
        if minX <= x[i] < maxX and minY <= y[i] < maxY:
            gridX = int(mapper(x[i], minX, maxX, 0, xLength))
            gridY = int(mapper(y[i], minY, maxY, yLength, 0))
            if value.any():
                spcaialFreq[gridY][gridX] += value[i]
            else:
                spcaialFreq[gridY][gridX] += 1
    if grid:
        return spcaialFreq, lon, lat
    else:
        return spcaialFreq


def t_test(a, b):
    '''
    两个正态总体的t双侧检验，计算t值
    :param a: list
    :param b: list
    :return:t value
    '''
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    na = len(a)
    nb = len(b)
    sa = np.var(a, axis=0)
    sb = np.var(b, axis=0)
    avga = np.mean(a, axis=0)
    avgb = np.mean(b, axis=0)
    t = (avga-avgb)/np.sqrt((na*sa)+(nb*sb)/(na+nb-2)*(1/na+1/nb))
    return t


def ols(x, y):
    '''
    一元线性最小二乘法回归，输入x与y数组，返回b,b0,r
    :param x: list or array
    :param y: list or array
    :return: [b,b0,r]
    '''
    import numpy as np
    if len(x) != len(y):
        raise Exception("x and y not the same length!")
    x = np.array(x)
    y = np.array(y)
    b = (sum(x*y)-((np.sum(x)*np.sum(y))/len(x))) / \
        (np.sum(x**2)-np.sum(x)**2/len(x))
    b0 = np.mean(y)-b*np.mean(x)
    yh = x*b+b0
    r = np.sum((yh-np.mean(y))**2)/np.sum((y-np.mean(y))**2)
    return [b, b0, r]


def moving_avg(array, n, bn=False):
    '''
    计算序列的滑动平均值
    :param array: list or nparray 输入一维序列
    :param n: int 数据滑动平均的数量，奇数
    :param bn: bool 是否补全滑动平均的头和尾(使用n-2, n-4, ...滑动平均计算)
    :return: list 滑动平均后数组
    '''
    import numpy as np
    data = []
    l = len(array)
    if bn:
        for i in range(l):
            if i < int(n/2):
                data.append(np.mean(array[0:i*2+1]))
            elif i >= l-int(n/2):
                data.append(np.mean(array[i-(l-i-1):l]))
            else:
                data.append(np.mean(array[i-int(n/2):i+int(n/2)]))
    else:
        for i in range(l-(n-1)):
            data.append(np.mean(array[i:i+n]))

    return data


def Gaussian(array, karray):
    '''
    高斯消元法，消列计算
    :param array: 数组方阵
    :param k: 数组，消去的列,从0开始
    :return: 结果方阵
    '''
    a = array
    for k in karray:
        for i in range(len(array)):
            for j in range(len(array[i])):
                if i == k and j != k:
                    array[i][j] = -(a[k][j]/a[k][k])
                elif i != k and j != k:
                    array[i][j] = a[i][j] - (a[k][j]*a[i][k])/a[k][k]
                elif i == k and j == k:
                    array[i][j] = 1/a[k][k]
                elif i != k and j == k:
                    array[i][j] = -(a[i][k]/a[k][k])
        a = array
    return array


def information_diffusion(y, u):
    '''
    信息扩散法计算
    :param y: list 观测值
    :param u: list 论域
    :return: list 各个论域值的概率
    '''
    b = max(y)
    a = min(y)
    m = len(y)

    # Calculate h
    if m < 5:
        raise Exception("Too less examples!")
    elif m == 5:
        h = 0.8146 * (b - a)
    elif m == 6:
        h = 0.5690 * (b - a)
    elif m == 7:
        h = 0.4560 * (b - a)
    elif m == 8:
        h = 0.3860 * (b - a)
    elif m == 9:
        h = 0.3362 * (b - a)
    elif m == 10:
        h = 0.2986 * (b - a)
    else:
        h = 1.4208 * (b - a) / (m - 1)
    print("h:", h)
    f = []
    for j in y:
        ff = []
        for i in u:
            ff.append(2.718281**(-1 * ((j-i)**2)/(2*h**2)) /
                      (h*(2*3.1415926)**0.5))
        f.append(ff)

    g = []
    for i in f:
        gg = []
        fsum = sum(i)
        for j in i:
            gg.append(j/fsum)
        g.append(gg)

    q = []
    for i in range(len(u)):
        qq = []
        for j in range(m):
            qq.append(g[j][i])
        q.append(sum(qq))
    p = [i/sum(q) for i in q]
    return p

class AHP:
    RI = [0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,
    1.49,1.52,1.54,1.56,1.58,1.59]

    def __init__(self,m):
        '''
        层次分析法
        :param m: 判断矩阵
        '''
        import numpy as np
        n = len(m)
        Mi = []
        for i in range(n):
            mm = 1
            for j in range(n):
                mm *= m[i][j]
            Mi.append(mm)
        wb = [i**(1/n) for i in Mi]
        wi = sum(wb)
        w = np.array([i/wi for i in wb])
        self.weight = w
        l = np.dot(m,w)
        l2 = l/w/n
        lmax = sum(l2)
        self.lambda_Max = lmax
        self.CI = (lmax - n)/(n - 1)
        self.CR = self.CI / self.RI[n-1]


class GRA:
    def std(self,array,t):
        mi = min(array)
        ma = max(array)
        if t == 1:
            data = [(i-mi)/(ma-mi) for i in array]
        elif t == 2:
            data = [(ma - i)/(ma - mi) for i in array]
        elif t == 3:
            data = [i/array[0] for i in array]
        return data

    def __init__(self,m,rho=0.5,t=3,omega=None):
        '''
        灰色关联度
        :param m: 数据矩阵，第一行为目标相关值，其余为相关因子值
        :param t: 标准化种类
        :param rho: rho值，常取0.5
        :param omega: None or list, omega相关权重，None为平均权重
        '''
        import numpy as np
        l = len(m)
        k = len(m[0])
        data = np.array([self.std(m[i],t) for i in range(l)])
        diff = np.array([np.abs(data[0] - data[i+1]) for i in range(l-1)])
        ma = np.max(diff)
        mi = np.min(diff)
        r0 = np.array([(mi + rho * ma)/(diff[i] + rho * ma) for i in range(l-1)])
        if omega == None:
            omega = 1 / k
            r = [omega * sum(r0[i]) for i in range(l-1)]
        else:
            r = [omega[i] * sum(r0[i]) for i in range(l - 1)]
        self.relate = r
        self.weight = np.array(r) / sum(r)


class standardized:

    def std(self,array,r=None):
        '''
        标准化数列，距平/方差
        :param array:输入数列
        :param r:四舍五入位数(round)
        :return:标准化序列
        '''
        import math
        avg = sum(array) / len(array)
        ano = [i - avg for i in array]
        std = math.sqrt(sum([i ** 2 for i in ano]) / (len(array) - 1))
        if r:
            return [round(i / std, r) for i in ano]
        else:
            return [i / std for i in ano]

    # 正向指标
    def p_indicator(self,array):
        mi = min(array)
        ma = max(array)
        return [(i - mi) / (ma - mi)for i in array]

    # 负向指标
    def n_indicator(self, array):
        mi = min(array)
        ma = max(array)
        return [(ma - i) / (ma - mi) for i in array]

def EWM(array,stdType="p"):
    '''
    熵权法计算权重
    :param array: 数据二维矩阵
    :param stdType: 标准化算法,"p"为正向指标,"n"为负向指标
    :return: [w,s] w为权重list，s为评分list
    '''
    import numpy as np
    std = standardized()
    if stdType == "p":
        data = [std.p_indicator(i) for i in array]
    elif stdType == "n":
        data = [std.n_indicator(i) for i in array]

    data = np.array(data)
    data[data == 0] = 0.00001
    p = np.array([i/np.sum(i) for i in data])
    e = np.array([(-1/np.log(len(data[0]))) * sum(i*np.log(i)) for i in p])
    d = 1-e
    w = d/sum(d)
    s = [sum(data[i]*w[i]) for i in range(len(data))]
    return [w,s]
