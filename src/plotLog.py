import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('dqn.log')
#df = df['stock']#,'credit','max_credit','sell','hold','buy']
print(df.columns.values.tolist())
# df = df.iloc(:,0)
print(df.iloc[:,2:])

step = df.iloc[:,2].values
stock = df.iloc[:,3].values
credit = df.iloc[:,4].values
max_credit = df.iloc[:,5].values
sell = df.iloc[:,6].values
hold = df.iloc[:,7].values
buy = df.iloc[:,8].values
print(step)

plt.plot(step, credit)
# plt.plot(step, stock*100)
plt.plot(step, max_credit)
# plt.plot(step[::10], sell[::10],'r')
# plt.plot(step[::10], hold[::10],'b')
# plt.plot(step[::10], buy[::10], 'g')
plt.show()
