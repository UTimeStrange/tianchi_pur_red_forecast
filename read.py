#!/usr/bin/env python
# encoding: utf-8
"""
@author: strangeli
@contact: strangeli@tencent.com
@software: pycharm
@file: read.py
@time: 2020/6/9 19:17
@desc:
"""
import pandas as pd


def date_parse(dates):
    return pd.datetime.strptime(dates, '%Y%m%d')


def generate_seq():
    user_balance = pd.read_csv('./prdata/user_balance_table.csv', parse_dates=['report_date'],
                               index_col='report_date', date_parser=date_parse)
    df_purchase = user_balance.groupby(['report_date'])['total_purchase_amt'].sum()
    df_redeem = user_balance.groupby(['report_date'])['total_redeem_amt'].sum()

    purchase_seq = pd.Series(df_purchase, name='y')
    redeem_seq = pd.Series(df_redeem, name='y')
    purchase_seq.to_csv('./shaped_data/purchase_seq.csv', header=True, encoding='utf-8')
    redeem_seq.to_csv('./shaped_data/redeem_seq.csv', header=True, encoding='utf-8')


if __name__ == '__main__':
    generate_seq()






