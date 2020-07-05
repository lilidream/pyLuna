def standardized(array, r=None):
    '''
    标准化变量函数
    :param array:(list)一组需要标准化操作的数组
    :param r: (int)四舍五入保留r位小数，不填则完全返回
    :return: (list)标准化操作后的数据
    '''
    import math
    avg = sum(array) / len(array)
    ano = [i - avg for i in array]
    std = math.sqrt(sum([i ** 2 for i in ano]) / (len(array) - 1))
    if r:
        return [round(i / std, r) for i in ano]
    else:
        return [i / std for i in ano]


def mapper(x, x1, x2, X1, X2):
    '''
    一维线性映射函数，返回在x1-x2中的x，映射到X1-X2后的值。
    :return: float
    '''
    return ((X2 - X1) * (x - x1) / (x2 - x1)) + X1


def mapper_array(array, x1, x2, X1, X2):
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
    计算矩阵序列双变量t统计量值
    :param a: 以时间为第一维度的平面数据矩阵
    :param b: 同上
    :return:
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
    b = max(y)
    a = min(y)
    m = len(y)

    # Calculate h
    if m < 5:
        raise "Too less examples!"
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
