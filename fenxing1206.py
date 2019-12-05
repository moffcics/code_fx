# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_finance as mpf
from base.plot_K_Line import plot_k_line
import pysnooper

data_dir = '../data/'
tmp_dir = "../tmp/"


# 判断相邻的两根K线的关系：向上、向下、包含
# @pysnooper.snoop()
def contain_process(df):  #定义包含处理的类
    rows = df.shape[0]    #取原始数据的行数值
    res_df = pd.DataFrame()  # 存放分型结果
    temp_df = df[0:1]  # 取第一行
    trend = []  # 存放K线的走势：1-向上 2-向下 3-包含
    for i in range(1, rows):
        temp_df = temp_df.reset_index(drop=True) #梳理原来的index，使它变成dataframe可以使用的顺序index
        case1 = temp_df.high[0] < df.high[i] and temp_df.low[0] < df.low[i]  # 向上趋势
        case2 = temp_df.high[0] > df.high[i] and temp_df.low[0] > df.low[i]  # 向下趋势
        case3 = temp_df.high[0] == df.high[i] and temp_df.low[0] == df.low[i]  # 高低相等,相互包含
        # 包含判断
        case3_1 = temp_df.high[0] > df.high[i] and temp_df.low[0] == df.low[i]  # 左包含:平底
        case3_2 = temp_df.high[0] > df.high[i] and temp_df.low[0] < df.low[i]  # 左包含:中间
        case3_3 = temp_df.high[0] == df.high[i] and temp_df.low[0] < df.low[i]  # 左包含:平顶
        case3_4 = temp_df.high[0] < df.high[i] and temp_df.low[0] == df.low[i]  # 右包含:平底
        case3_5 = temp_df.high[0] < df.high[i] and temp_df.low[0] > df.low[i]  # 右包含:居中
        case3_6 = temp_df.high[0] == df.high[i] and temp_df.low[0] > df.low[i]  # 右包含:平顶

        if case1:  # 向上趋势
            trend.append(1)
            res_df = pd.concat([res_df, temp_df], axis=0) #K线数据存入dataframe
            temp_df = df[i:i + 1] #开始下一行的判断
        elif case2:  # 向下趋势
            trend.append(2)
            res_df = pd.concat([res_df, temp_df], axis=0)
            temp_df = df[i:i + 1]
        elif case3:  # 高低相等，相互包含
            trend.append(3)

        # 左包含和右包含需要区分上升趋势还是下降趋势
        elif case3_1 or case3_2 or case3_3:
            if trend[-1] == 1:  # 上升趋势下的左包含
                temp_df.low[0] = df.low[i] #取右侧K线的低点作为合并后k线的低点，即低低取高原则
            elif trend[-1] == 2:  # 下降趋势下的左包含
                temp_df.high[0] = df.high[i]
            else:
                print("---Exception---")

        elif case3_4 or case3_5 or case3_6:
            if trend[-1] == 1:  # 上升趋势下的右包含
                temp_df.high[0] = df.high[i]
            elif trend[-1] == 2:  # 下降趋势下的右
                temp_df.low[0] = df.low[i]
            else:
                print("====Exception====")
    print(res_df.head())
    res_df.to_csv(tmp_dir + "fenxing_res.csv", index=False)
    return res_df


# 画出包含处理后的效果图
def plot_after_contain_process(res_df):
    # 因为使用candlestick2函数，要求输入open、close、high、low,为了美观，处理k线的最大最小值、开盘收盘价，之后k线不显示影线。
    for i in range(len(res_df)):
        if res_df.open[i] > res_df.close[i]:
            res_df.open[i] = res_df.high[i]
            res_df.close[i] = res_df.low[i]
        else:
            res_df.open[i] = res_df.low[i]
            res_df.close[i] = res_df.high[i]
    # 画出k线图
    plot_k_line(res_df, title_note="fenxing_res")


# 判断分型结构：顶分型、底分型,保存分型结构的类型和分型的三个元素的极值(6个数)
# @pysnooper.snoop()
def find_fenxing(df): #这里的df数据需要使用包含处理后的dataframe
    rows = df.shape[0]
    fenxing_data = []
    i = 2
    while i < rows:
        # 顶分型
        case1 = df.high[i - 2] < df.high[i - 1] and df.high[i - 1] > df.high[i] \
                and df.low[i - 2] < df.low[i - 1] and df.low[i - 1] > df.low[i]
        # 底分型
        case2 = df.high[i - 1] < df.high[i - 2] and df.high[i - 1] < df.high[i] \
                and df.low[i - 1] < df.low[i - 2] and df.low[i - 1] < df.low[i]
        # 上升
        case3 = df.high[i - 2] < df.high[i - 1] < df.high[i] \
                and df.low[i - 2] < df.low[i - 1] < df.low[i]
        # 下降
        case4 = df.high[i - 2] > df.high[i - 1] > df.high[i] \
                and df.low[i - 2] > df.low[i - 1] > df.low[i]
        if case1:
            print("顶分型")
            item = [df['trade_date'][i - 1], 'up', df['high'][i - 2], df['low'][i - 2], #trade_date是tushare导入时候的交易日期写法
                    df['high'][i - 1], df['low'][i - 1], df['high'][i], df['low'][i]]
            fenxing_data.append(item)
        elif case2:
            print("底分型")
            item = [df['trade_date'][i - 1], 'down', df['high'][i - 2], df['low'][i - 2],
                    df['high'][i - 1], df['low'][i - 1], df['high'][i], df['low'][i]]
            fenxing_data.append(item)
        elif case3:
            print("上升")
        elif case4:
            print("下降")
        else:
            print("=======Exception========")
        i += 1
    fenxing_df = pd.DataFrame(fenxing_data, columns=['trade_date', 'fenxing_type',
                                                     'e1_max', 'e1_min', 'e2_max', 'e2_min', 'e3_max', 'e3_min'])
    fenxing_df.to_csv(tmp_dir + "find_fenxing.csv", index=False)
    # print(fenxing_df.head())
    return fenxing_df


# 判断笔的成立条件
def stroke(kline_df, kline_proed_df, fenxing_df):
    '''
    :param kline_df: 原始的K线数据DataFrame
    :param kline_proed_df: 对K线数据进行包含处理后的DataFrame
    :param fenxing_df: 分型结构的数据DataFrame
    :return:
    '''
    # 1.相邻两个分型彼此的第二元素之间k线大于5根（包括第二元素）
    # 2.一笔必须是顶底相连
    # 3.相邻两个分型的元素不能共用
    # 4.后一分型的第二元素极值不能在前一分型的第一元素范围内
    # 5. 跳空范围在前一笔的极值外则跳空成为一笔
    rows = fenxing_df.shape[0] #取分型状态数据表行数
    temp_df=fenxing_df[1:2]  #取第二行为初始启动对象
    stroke_data=[]
    for i in range(2,rows-1):
        d0,d1,d2=temp_df['trade_date'].values[0],temp_df['trade_date'].values[1],fenxing_df.loc[i,'trade_date'] #取第一、第二，三个分型的交易日期
        type0,type1,type2=temp_df['fenxing_type'].values[0],temp_df['fenxing_type'].values[1],fenxing_df.loc[i,'fenxing_type']#取取第一、第二，三个分型的类型
        span_num=kline_df[kline_df['trade_date']==d2].index-kline_df[kline_df['trade_date']==d1].index# 取第二，三个分型交易日之间的交易天数
        span_num_proed=kline_proed_df[kline_df['trade_date']==d2].index-kline_df[kline_df['trade_date']==d1].index# 取第二，三个分型包含处理后的K线数量
        prefirst_e2_index=kline_proed_df[kline_proed_df['trade_date']==d1-1].index #取第一个分型（包含处理后）的索引
        first_e2_index=kline_proed_df[kline_proed_df['trade_date']==d1].index #取第二个分型的第二元素（包含处理后）的索引
        second_e2_index = kline_proed_df[kline_proed_df['trade_date'] == d2].index#取第三个分型的第二元素（包含处理后）的索引
        first_e3_d=kline_proed_df.loc[first_e2_index+1,'trade_date']#取第二个分型的第三元素（包含处理后）的交易日期
        second_e1_d=kline_proed_df.loc[second_e2_index-1,'trade_date']#取第三个分型的第一元素（包含处理后）的交易日期
        second_e2_d=kline_proed_df.loc[second_e2_index,'trade_date']  #取第三个分型的第二元素（包含处理后）的交易日期
        prefirst_e2_d=kline_proed_df.loc[]
        second_e2_min,second_e2_max=kline_proed_df.loc[second_e2_index,'low'],kline_proed_df.loc[second_e2_index,'high'] #取第三个分型的第二元素的极值（包含处理后）
        first_e1_min,first_e1_max=kline_proed_df.loc[first_e2_index-1,'low'],kline_proed_df.loc[first_e2_index-1,'high'] #取第二个分型的第一元素的极值（包含处理后）
        prefirst_e2_max,prefisrt_e2_min=kline_proed_df.loc[first_e2_index] #取第一个分型的第二元素的极值（包含处理后）
        if type1!=type2:
            if second_e2_max=<first_e1_max and second_e2_min>=first_e1_min: #判断分型元素是否包含
                i+=1 #包含，则与下一个分型去比较进行判断
            else:
                if span_num>=5:
                    if span_num_proed>=5#case1 包含处理后的K线有5个,严笔
                        if second_e2_min>first_e1_min and second_e2_max>first_e1_max:
                            print('向上一笔')
                            print(first_e1_min, second_e2_max)#笔的极值应该取该顶底分型日期内所有K线范围内的的极值，这里需要优化
                            '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_min,second_e2_max]
                            stroke_data.append(item)'''
                        else:
                            print('向下一笔')
                            print(first_e1_max, second_e2_min)  # 笔的极值应该取该顶底分型日期内所有K线范围内的的极值，这里需要优化
                            '''item=[fenxing_df['trade_date'][1],'down',fenxing_df['trade_date'][2],first_e2_max,second_e2_min]
                            stroke_data.append(item)'''
                    if span_num_proed=4#case2 包含处理后的K线有4个，宽笔
                        if second_e2_min>first_e1_min and second_e2_max>first_e1_max:
                            print('向上一笔')
                            print(first_e1_min, second_e2_max)#笔的极值应该取该顶底分型日期内所有K线间的的极值，这里需要优化
                            '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_min,second_e2_max]
                            stroke_data.append(item)'''
                        else:
                            print('向下一笔')
                            print(first_e1_max, second_e2_min)#笔的极值应该取该顶底分型日期内所有K线间的的极值，这里需要优化
                            '''item=[fenxing_df['trade_date'][1],'down',fenxing_df['trade_date'][2],first_e2_max,second_e2_min]
                            stroke_data.append(item)'''
                    if span_num_proed=3#case3 包含处理后的K线有3个
                        for a in (1,span_num-1):
                            if kline_df['trade_date'==d1+a].low>prefirst_e2_max:
                                print(first_e2_min, second_e2_max) #向上跳空破前一分型第二元素的极值，向上一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_min,second_e2_max]
                                stroke_data.append(item)'''
                                break
                            if kline_df['trade_date'==d1+a].high< prefirst_e2_min:#包含处理后的第2分型（e1或e2)所在的那个原始K线的极值跳空突破前面分型元素的极值则一笔成立，反之不成立
                                print(first_e2_max, second_e2_min) #向下跳空破前一分型第二元素的极值，向下一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'down',fenxing_df['trade_date'][2],first_e2_max,second_e2_min]
                                stroke_data.append(item)'''
                                break
                            else:
                                i+=1 #没有跳空破极值，选择下一个分型重新开始判断
                    if span_num_proed=2#case4 包含处理后的K线有3个
                        for a in (1,span_num-1):
                            if kline_df['trade_date'==d1+a].low>prefirst_e2_max:
                                print(first_e2_min, second_e2_max) #向上跳空破前一分型第二元素的极值，向上一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_min,second_e2_max]
                                stroke_data.append(item)'''
                                break
                            if kline_df['trade_date'==d1+a].high< prefirst_e2_min:#包含处理后的第2分型（e1或e2)所在的那个原始K线的极值跳空突破前面分型元素的极值则一笔成立，反之不成立
                                print(first_e2_max, second_e2_min) #向下跳空破前一分型第二元素的极值，向下一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'down',fenxing_df['trade_date'][2],first_e2_max,second_e2_min]
                                stroke_data.append(item)'''
                                break
                            else:
                                i+=1 #没有跳空破极值，选择下一个分型重新开始判断
                else:
                    for a in (1,span_num-1):
                            if kline_df['trade_date'==d1+a].low>prefirst_e2_max:
                                print(first_e2_min, second_e2_max) #向上跳空破前一分型第二元素的极值，向上一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_min,second_e2_max]
                                stroke_data.append(item)'''
                                break
                            elif kline_df['trade_date'==d1+a].high< prefirst_e2_min:#包含处理后的第2分型（e1或e2)所在的那个原始K线的极值跳空突破前面分型元素的极值则一笔成立，反之不成立
                                print(first_e2_max, second_e2_min) #向下跳空破前一分型第二元素的极值，向下一笔成立，宽笔
                                '''item=[fenxing_df['trade_date'][1],'up',fenxing_df['trade_date'][2],first_e2_max,second_e2_min]
                                stroke_data.append(item)'''
                                break
                            else:
                                i+=1 #没有跳空破极值，选择下一个分型重新开始判断
        else:
            i+=1 #两个分型类型一致，选择下一个与之比较
    stroke_df = pd.DataFrame(stroke_data, columns=['trade_date', 'stroke_type','first_e2', 'second_e2'])
    stroke_df.to_csv(tmp_dir + "stroke.csv", index=False)
    return stroke_df


if __name__ == '__main__':
    # test_data = pd.read_csv(data_dir + 'test_data.csv')
    # 对K线进行包含处理
    # kline_proed_df=contain_process(test_data)

    # res_data = pd.read_csv(tmp_dir + "fenxing_res.csv")
    # plot_after_contain_process(res_data)

    # 判断分型结构
    # fenxing_df=find_fenxing(res_data)



    # 判断笔
    kline_df=pd.read_csv(data_dir + 'test_data.csv')
    kline_proed_df=pd.read_csv(tmp_dir + "fenxing_res.csv")
    fenxing_df=pd.read_csv(tmp_dir + "find_fenxing.csv")
    stroke(kline_df, kline_proed_df, fenxing_df)
