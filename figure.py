# Matplotlib画图相关函数

def font(fontname,fontsize=12):
    '''
    在Matplotlib绘图中，导入自定义字体
    :param fontname: 字体名字，合法值有[wenquanyi]
    '''
    from matplotlib.font_manager import FontProperties
    if fontname == "wenquanyi":
        font = FontProperties(fname="E:/pyLuna/data/Font/文泉驿微米黑.ttf",size=fontsize)
    else:
        raise("No this fontFamily!")
    return font
