import matplotlib.pyplot as plt
import pandas as pd


user_balance = pd.read_csv('./prdata/user_balance_table.csv')
df_tmp = user_balance.groupby(['report_date'])['total_purchase_amt', 'total_redeem_amt'].sum()
df_tmp.reset_index(inplace=True)

df_tmp['report_date'] = pd.to_datetime(df_tmp['report_date'], format='%Y%m%d')
df_tmp.index = df_tmp['report_date']

total_purchase_amt = plt.figure(figsize=(10, 4))
ax = total_purchase_amt.add_subplot(111)
ax.set(title='Total_Purchase_Amt',
       ylabel='Unit (yuan)', xlabel='Date')
plt.plot(df_tmp['report_date'], df_tmp['total_purchase_amt'])
plt.show()
