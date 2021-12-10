import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.ExcelFile('warn.xlsx')
df = df.parse(df.sheet_names[0])
dfd = df.to_numpy()

fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(6,8), dpi= 80)
plot_acf(df['Unnamed: 9'].tolist(), ax=ax1, lags=50)
plot_pacf(df['Unnamed: 9'].tolist(), ax=ax2, lags=20)

ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)
ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)
ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)
ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)

ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.show()

plt.plot(df['Unnamed: 9'][600:650], linewidth = 2, label='no_lag')
plt.plot(df['Unnamed: 9'].shift(3)[600:650], linestyle = ':', label='lag_1')
plt.plot(df['Unnamed: 9'].shift(6)[600:650], linestyle = '--', label='lag_2')
plt.title('Example of lags')
plt.legend()
plt.show()
