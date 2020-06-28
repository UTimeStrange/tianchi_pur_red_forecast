import pandas as pd


def if_2_or_3(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 1 or date.weekday() == 2 or date.weekday() == 3:
        return 1
    else:
        return 0


def if_6_or_7(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 0 or date.weekday() == 6:
        return 1
    else:
        return 0


def get_123_days():
    tmp = pd.DataFrame()
    tmp['ds'] = pd.date_range(start=pd.datetime(2014, 4, 1), end=pd.datetime(2014, 10, 1), freq='D')
    tmp['filter'] = tmp['ds'].apply(if_2_or_3)
    res = tmp[tmp['filter'] == 1]['ds'].values
    return res


def get_67_days():
    tmp = pd.DataFrame()
    tmp['ds'] = pd.date_range(start=pd.datetime(2014, 4, 1), end=pd.datetime(2014, 10, 1), freq='D')
    tmp['filter'] = tmp['ds'].apply(if_6_or_7)
    res = tmp[tmp['filter'] == 1]['ds'].values
    return res


week123 = pd.DataFrame({
  'holiday': 'week123',
  'ds': pd.to_datetime(get_123_days()),
  'lower_window': 0,
  'upper_window': 1,
})

week67 = pd.DataFrame({
  'holiday': 'week67',
  'ds': pd.to_datetime(get_67_days()),
  'lower_window': 0,
  'upper_window': 1,
})


holidays = pd.concat((week123, week67))


if __name__ == '__main__':
    # xx =  pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
    #                     '2010-01-24', '2010-02-07', '2011-01-08',
    #                     '2013-01-12', '2014-01-12', '2014-01-19',
    #                     '2014-02-02', '2015-01-11', '2016-01-17',
    #                     '2016-01-24', '2016-02-07'])
    # print(xx)
    # print(type(xx))
    # exit()

    print(week123['ds'])
    print(week67['ds'])

