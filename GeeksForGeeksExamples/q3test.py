# Example of an Autoregression Model taken from GeeksForGeeks
# Trained on the Weather Data .csv file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Weather_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df["Temperature"] = df["Temperature"].interpolate()

result = adfuller(df["Temperature"])
print(f"ADF statistic = {result[0]:.3f}, p-value = {result[1]:.3f}")

df_diff = df["Temperature"].diff().dropna()

fig, ax = plt.subplots(2,1, figsize=(10,6))
plot_acf(df_diff, lags=30, ax=ax[0])
plot_pacf(df_diff, lags=30, ax=ax[1])
plt.tight_layout()
plt.show()

n = len(df_diff)
train_end = int(n * 0.8)
train = df_diff.iloc[:train_end]
test = df_diff.iloc[train_end:]

p = 13
model = AutoReg(train, lags=p, old_names=False)
model_fit = model.fit()
print(model_fit.summary())

pred_train = model_fit.predict(start=train.index[p], end=train.index[-1], dynamic=False)
pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)

rmse = np.sqrt(mean_squared_error(test, pred_test))
mae = mean_absolute_error(test, pred_test)
print(rmse, mae,sep="\n")

last_train_value = df["Temperature"].iloc[train_end]
forecast_orig = pred_test.cumsum() + last_train_value
forecast_orig.index = test.index

plt.figure(figsize=(12,5))
plt.plot(df["Temperature"], label="Observed", linewidth=1)
plt.plot(forecast_orig, label="AR forecast", linestyle="--")
plt.axvline(df.index[train_end], alpha=0.5, linestyle=":")
plt.legend()
plt.show()