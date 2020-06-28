import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from the_log import Logger
import utils

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf, suppress=True)


def fun(df: pd.DataFrame):
    peek_days, peek_value, peek_day_of_week = [], [], []
    low_days, low_value, low_day_of_week = [], [], []
    for i in range(1, len(df) - 1):
        if df['y'].values[i] > df['y'].values[i-1] and df['y'].values[i] > df['y'].values[i+1]:
            peek_days.append(df['ds'].values[i])
            peek_value.append(df['y'].values[i])
            peek_day_of_week.append(df['day_of_week'].values[i])
        if df['y'].values[i] < df['y'].values[i-1] and df['y'].values[i] < df['y'].values[i+1]:
            low_days.append(df['ds'].values[i])
            low_value.append(df['y'].values[i])
            low_day_of_week.append(df['day_of_week'].values[i])
    return peek_days, peek_value, peek_day_of_week, low_days, low_value, low_day_of_week


if __name__ == '__main__':
    purchase = pd.read_csv('./shaped_data/purchase_seq.csv', encoding='utf-8').rename(columns={'report_date': 'ds'})
    redeem = pd.read_csv('./shaped_data/redeem_seq.csv', encoding='utf-8').rename(columns={'report_date': 'ds'})
    purchase['ds'] = pd.to_datetime(purchase['ds'], format='%Y-%m-%d')
    redeem['ds'] = pd.to_datetime(redeem['ds'], format='%Y-%m-%d')

    purchase['day_of_week'] = purchase['ds'].dt.dayofweek
    redeem['day_of_week'] = redeem['ds'].dt.dayofweek

    time_cut = pd.datetime(2014, 4, 1)
    purchase = purchase[purchase['ds'] >= time_cut]
    redeem = redeem[redeem['ds'] >= time_cut]

    # peek_days, peek_value, peek_day_of_week, low_days, low_value, low_day_of_week = fun(purchase)
    peek_days, peek_value, peek_day_of_week, low_days, low_value, low_day_of_week = fun(redeem)

    plt.figure(figsize=[19, 10], dpi=300)
    plt.plot(redeem['ds'], redeem['y'], label='redeem', color='blue')
    plt.scatter(x=peek_days, y=peek_value)
    plt.scatter(x=low_days, y=low_value)
    for i in range(len(peek_days)):
        plt.annotate("周{}".format(peek_day_of_week[i] + 1), xy=(peek_days[i], peek_value[i]),
                     xytext=(peek_days[i], peek_value[i]), size=20)
    for i in range(len(low_days)):
        plt.annotate("周{}".format(low_day_of_week[i] + 1), xy=(low_days[i], low_value[i]),
                     xytext=(low_days[i], low_value[i]), size=20)
    plt.xlabel('time')
    plt.ylabel('y')
    plt.xticks(rotation=30)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("./pics/redeem_看看峰值和谷值.png")
    pass