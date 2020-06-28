#!/usr/bin/env python
# encoding: utf-8
"""
@author: strangeli
@contact: strangeli@tencent.com
@software: pycharm
@file: prophet_train.py
@time: 2020/6/19 18:43
@desc:
"""

from tqdm import tqdm
from fbprophet import Prophet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from the_log import Logger
import utils


CAP_VALUE = 2.5

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf, suppress=True)


def get_model(params):
    """
    获取模型 主要是交叉验证需要重新训练模型 为了封装方便提取出来
    :param params:
    :return:
    """
    model = Prophet(**params)
    # 添加法定节假日
    model.add_country_holidays(country_name='CN')
    # 每个月月初和月末也不同
    # model.add_seasonality(name='two_weeks', period=14, fourier_order=2)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
    # 添加周二周三效应
    # model.add_regressor('if_2_or_3')
    # model.add_regressor('if_6_or_7')
    return model


def train(data: pd.DataFrame, params: dict, description: str, cross_validation=True, forecast_future=False,
          plot_lines=False, param_description="", my_logger=None, cap=None):
    """
    :param data: 数据集，要求时间列名为 ds, 目标序列列名为 y
    :param params: prophet 参数 要求传入一个字典
    :param description: 描述 要求为 string
    :param cross_validation: 交叉验证判断效果
    :param forecast_future: 预测未来数据
    :param plot_lines: 画出曲线图
    :param param_description: 参数描述
    :param my_logger:
    :param cap:
    :return:
    """
    all_return_acs = []
    cur_param = params['changepoint_prior_scale']
    valid_lens = []
    data['if_2_or_3'] = data['ds'].apply(utils.if_2_or_3)
    data['if_6_or_7'] = data['ds'].apply(utils.if_6_or_7)

    if cross_validation:
        for valid_len in [10, 30]:
            model = get_model(params=params)
            my_logger.logger.info("\n valid len = {}\n{}".format(valid_len, param_description))
            train_set = data[: -valid_len]
            valid_set = data[-valid_len:]

            model.fit(train_set)
            train_pred = model.predict()

            valid_2b_forecast = model.make_future_dataframe(valid_len)
            valid_2b_forecast['if_2_or_3'] = valid_2b_forecast['ds'].apply(utils.if_2_or_3)
            valid_2b_forecast['if_6_or_7'] = valid_2b_forecast['ds'].apply(utils.if_6_or_7)
            if cap:
                valid_2b_forecast['cap'] = cap

            train_and_valid_pred = model.predict(valid_2b_forecast)
            valid_pred = train_and_valid_pred[-valid_len:]

            all_valid_acc = valid_pred['yhat'].values.sum() / valid_set['y'].values.sum()
            all_train_acc = train_pred['yhat'].values.sum() / train_set['y'].values.sum()

            # 用于调参 返回结果
            all_return_acs.append(all_valid_acc)
            valid_lens.append(valid_len)

            my_logger.logger.info("\n{}长度训练集总准确率 = {}".format(len(train_set), all_train_acc))
            my_logger.logger.info("\n{}长度验证集总准确率 = {}".format(valid_len, all_valid_acc))

            month_need = valid_set['ds'].dt.month.unique()

            for m in month_need:
                m_month_data = valid_pred[valid_pred['ds'].dt.month == m]
                if len(m_month_data) >= 15:
                    tmp = data[(data['ds'] >= m_month_data['ds'].values[0]) & (data['ds'] <= m_month_data['ds'].values[-1])]
                    this_month_acc = m_month_data['yhat'].values.sum() / tmp['y'].values.sum()
                    my_logger.logger.info("\n{}月({}天)总准确率 = {}".format(m, len(m_month_data), this_month_acc))

            if plot_lines:
                f, ax = plt.subplots(nrows=1, ncols=1, figsize=[19, 10], dpi=500)
                model.plot(train_and_valid_pred, ax=ax)
                ax.plot(data['ds'], data['y'], label='real', color='red', alpha=0.8)
                ax.legend(fontsize=20)
                ax.set_xlabel('预测验证集{}天'.format(valid_len), fontsize=20)
                ax.set_title("{}_{}".format(description, param_description))
                f.savefig("./pics/param_fine_tune/预测结果曲线{}_{}_{}.png".format(
                    description, param_description, valid_len))

                fig2 = model.plot_components(train_and_valid_pred)
                fig2.savefig("./pics/prophet_components/参数分解{}_{}_{}.png".format(
                    description, param_description, valid_len))

                # plt.show()

    if forecast_future:
        model = get_model(params=params)
        model.fit(data)

        future = model.make_future_dataframe(periods=30)
        future['if_2_or_3'] = future['ds'].apply(utils.if_2_or_3)
        future['if_6_or_7'] = future['ds'].apply(utils.if_6_or_7)
        if cap:
            future['cap'] = cap

        future_forecast = model.predict(future)
        ori_future_forecast = future_forecast
        future_forecast = future_forecast[~future_forecast['ds'].isin(data['ds'])]

        res = future_forecast
        my_logger.logger.info("\n forecasting 30 days future of {} data \n file saved".format(description))
        res.to_csv('./shaped_data/{}_future_forecast.csv'.format(description), index=False, encoding='utf-8')
        if plot_lines:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=[19, 10], dpi=300)
            model.plot(ori_future_forecast, ax=ax)
            ax.plot(data['ds'], data['y'], label='real', color='red', alpha=0.8)
            ax.legend(fontsize=20)
            ax.set_xlabel('预测未来30天', fontsize=20)
            ax.set_title("{}_{}".format(description, param_description))
            f.savefig("./pics/param_fine_tune/预测未来30天{}_{}.png".format(
                description, param_description))
    return all_return_acs, cur_param, valid_lens


def gen_res(res_description=""):
    pur = pd.read_csv("./shaped_data/purchase_future_forecast.csv", encoding='utf-8')
    red = pd.read_csv("./shaped_data/redeem_future_forecast.csv", encoding='utf-8')

    pur['ds'] = pd.to_datetime(pur['ds'], format='%Y-%m-%d')
    red['ds'] = pd.to_datetime(red['ds'], format='%Y-%m-%d')

    pur['day_of_week'] = pur['ds'].dt.dayofweek

    res = pd.DataFrame()
    res['ds'] = pur['ds'].apply(lambda x: x.year * 10000 + x.month * 100 + x.day).astype(int)
    res['pur'] = pur['yhat'].astype(int)

    idx_2_3 = pur['day_of_week'].isin([1, 3])
    idx_6_7 = pur['day_of_week'].isin([0, 6])
    res.loc[idx_2_3, 'pur'] = pur.loc[idx_2_3, 'yhat_upper']
    res.loc[idx_6_7, 'pur'] = pur.loc[idx_6_7, 'yhat_lower']

    res['red'] = red['yhat'].astype(int)
    res.to_csv("./all_res/res{}.csv".format(res_description), index=False, header=False, encoding='utf-8')


def pur_or_redeem_train(df, description, cps, cap=None):
    cur_log = Logger('./train.log', level='debug')
    param = {'growth': 'linear',
             'n_changepoints': 25,
             'changepoint_range': 0.8,
             'yearly_seasonality': False,
             'weekly_seasonality': False,
             'daily_seasonality': False,
             'seasonality_mode': 'additive',
             'holidays': utils.holidays,
             'seasonality_prior_scale': 200,
             'holidays_prior_scale': 200,
             'changepoint_prior_scale': cps}  # 0.02 黄钻best
    if cap:
        param['growth'] = 'logistic'
    final_res_dic = {}
    cur_log.logger.info("\n\n {} train start".format(description))
    for par in cps:
        param['changepoint_prior_scale'] = par
        valid_acs, cur_param, valid_lens = train(data=df,
                                                 params=param,
                                                 description=description,
                                                 cross_validation=True,
                                                 forecast_future=True,
                                                 plot_lines=True,
                                                 param_description='cps={}'.format(par),
                                                 my_logger=cur_log,
                                                 cap=cap,
                                                 )
        inner_dic = {}
        for i in range(len(valid_lens)):
            inner_dic['valid_len = {}'.format(valid_lens[i])] = valid_acs[i]
            final_res_dic['param = {}'.format(cur_param)] = inner_dic

    for key, value in final_res_dic.items():
        cur_log.logger.info("\n\n\n {}\n {}".format(key, json.dumps(value)))


if __name__ == '__main__':

    purchase = pd.read_csv('./shaped_data/purchase_seq.csv', encoding='utf-8').rename(columns={'report_date': 'ds'})
    purchase['ds'] = pd.to_datetime(purchase['ds'], format='%Y-%m-%d')
    purchase['day_of_week'] = purchase['ds'].dt.dayofweek

    redeem = pd.read_csv('./shaped_data/redeem_seq.csv', encoding='utf-8').rename(columns={'report_date': 'ds'})
    redeem['ds'] = pd.to_datetime(redeem['ds'], format='%Y-%m-%d')
    redeem['day_of_week'] = redeem['ds'].dt.dayofweek
    redeem['cap'] = CAP_VALUE * 10**8

    time_cut = pd.datetime(2014, 4, 1)
    purchase = purchase[purchase['ds'] >= time_cut]

    time_cut = pd.datetime(2014, 7, 1)
    redeem = redeem[redeem['ds'] >= time_cut]

    # pur_best = 0.0025
    # pur_or_redeem_train(purchase, 'purchase', cps=np.linspace(0.001, 0.1, 10))
    # pur_or_redeem_train(purchase, 'purchase', cps=[1.6])

    # redeem best = 0.09
    # pur_or_redeem_train(redeem, 'redeem', cps=[0.8], cap=CAP_VALUE * 10**8)

    # gen_res("周23取峰值67取谷值")


    pass
