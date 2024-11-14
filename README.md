# DEVELOPED BY : SHALINI K
# REGISTER NUMBER : 212222240095
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```PY
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the provided Bitcoin dataset
file_path = 'coin_Bitcoin.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Extract the 'Close' column for modeling
data_values = data['Close'].dropna().values

# 1. ARMA(1,1) Model for Bitcoin Close Prices

# Fit the ARMA(1,1) model
arma11_model = ARIMA(data_values, order=(1, 0, 1))
arma11_fit = arma11_model.fit()

# Plot the fitted ARMA(1,1) time series
plt.figure(figsize=(10, 6))
plt.plot(data_values, label='Original Data')
plt.plot(arma11_fit.fittedvalues, label='Fitted ARMA(1,1)', color='red')
plt.title('ARMA(1,1) Fitted Process - Bitcoin Close Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Display ACF and PACF plots for the actual data
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_values, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(data_values, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for Bitcoin Close Prices')
plt.tight_layout()
plt.show()

# 2. ARMA(2,2) Model for Bitcoin Close Prices

# Fit the ARMA(2,2) model
arma22_model = ARIMA(data_values, order=(2, 0, 2))
arma22_fit = arma22_model.fit()

# Plot the fitted ARMA(2,2) time series
plt.figure(figsize=(10, 6))
plt.plot(data_values, label='Original Data')
plt.plot(arma22_fit.fittedvalues, label='Fitted ARMA(2,2)', color='red')
plt.title('ARMA(2,2) Fitted Process - Bitcoin Close Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Display ACF and PACF plots for the actual data
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_values, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(data_values, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for Bitcoin Close Prices')
plt.tight_layout()
plt.show()

```

### OUTPUT:

<table>
  <tr>
    <td style="width:50%">
      <h3>SIMULATED ARMA(1,1) PROCESS:</h3>
      <img src="https://github.com/user-attachments/assets/d93ea978-68fb-4c6c-9842-ef449a0d5c93" style="width:48%; height:auto;">
    </td>
    <td style="width:50%">
      <h3>Partial Autocorrelation (ARMA 1,1):</h3>
      <img src="https://github.com/user-attachments/assets/680efae5-0627-4c37-9028-cc1f62f63bf9" style="width:48%; height:auto;">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <h3>SIMULATED ARMA(2,2) PROCESS:</h3>
      <img src="https://github.com/user-attachments/assets/b89eae64-d15d-42b6-b398-9c7a6d010a58" style="width:48%; height:auto;">
    </td>
    <td style="width:50%">
      <h3>Partial Autocorrelation (ARMA 2,2):</h3>
      <img src="https://github.com/user-attachments/assets/02523c49-0c19-45af-9c83-16d9e5f07ed9" style="width:48%; height:auto;">
    </td>
  </tr>
</table>



# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
