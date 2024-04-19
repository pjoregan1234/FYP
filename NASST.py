import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

exceldf = pd.read_excel('NASST.xlsx')

time_data = exceldf.iloc[:, 1]
temperature_data = exceldf.iloc[:, 2]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time_data, temperature_data, label='SST (Celcius)', color='blue')
plt.xlabel('Year', fontsize=14)
plt.ylabel('SST (Celcius)', fontsize=14)
plt.title('North Atlantic SST', fontsize=16)
plt.xlim(1993.03, 2024.214)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('NASST.png')
plt.show()

time = []
temp = []

print(exceldf)

current_year = exceldf["Int Year"][0]

for x in range(len(exceldf)):
    if (exceldf["Int Year"][x] == current_year):
        time.append(exceldf["Fraction Year"][x])
        temp.append(exceldf["Average Temperature"][x])
    else:
        if current_year == 2023:
            plt.plot(time, temp, color="red")
        if current_year == 2022:
            plt.plot(time, temp, color="blue")
        if current_year == 2012:
            plt.plot(time, temp, color="orange")
        if current_year == 2010:
            plt.plot(time, temp, color="green")
        if current_year <= 2021:
            plt.plot(time, temp, color="grey")
        current_year = current_year + 1
        time = []
        temp = []
plt.plot(time, temp, color="yellow")

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='2024'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='2023'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='2022'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='All Other Years since 1993')]

plt.xlabel('Time (Fraction of Year)')
plt.ylabel('Temperature (Celcius)')
plt.title('North Atlantic SST')
plt.legend(handles=legend_elements, title='Year', loc='upper left')
plt.tight_layout()
plt.xlim(0, 1)
plt.savefig('NASSTYearly.png')
plt.show()

X = time_data.to_numpy().reshape(-1, 1)
y = temperature_data.to_numpy()

# Split value
split_value = 2023
# Boolean indexing to split the array
X1 = X[X < split_value].reshape(-1,1)
X2 = X[X >= split_value].reshape(-1,1)
y1 = y[:len(X1)]
y2 = y[len(X1):]


rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y1.size), size=250, replace=False)
X_train, y_train = X1[training_indices], y1[training_indices]

plt.scatter(X_train, y_train)
plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF
long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)

from sklearn.gaussian_process.kernels import ExpSineSquared
seasonal_kernel = (2.0**2* RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed"))

from sklearn.gaussian_process.kernels import WhiteKernel
noise_kernel = WhiteKernel(
    noise_level=0.6, noise_level_bounds=(1e-3, 1e3))

kernel = (long_term_trend_kernel + seasonal_kernel + noise_kernel)

gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
print(gaussian_process.kernel_)


import datetime
import numpy as np
today = datetime.datetime.now()
current_month = today.year + today.month / 12
X_test = np.linspace(start=1993, stop=2025, num = 1000).reshape(-1, 1)
mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)

plt.plot(X1, y1, color="black", linestyle="solid", label="Measurements (Not Trained on)")
plt.plot(X2, y2, color="red", linestyle="solid", label="Measurements")
plt.plot(X_test, mean_prediction, color="tab:blue", alpha=0.4, label="Gaussian process")
plt.fill_between(
    X_test.ravel(),
    mean_prediction - 1.96*std_prediction,
    mean_prediction + 1.96*std_prediction,
    color="tab:blue",
    alpha=0.2,
)
plt.xlim(2020, 2025)
plt.legend()
plt.legend(fontsize='x-small')
plt.xlabel("Year")
plt.ylabel("Temperature (Celcius)")
plt.title("North Atlantic SST")
plt.savefig('NASST_GP.png', dpi=1000)
plt.show()

# Define input space
N = 100
X = np.linspace(0, 9, N).reshape(-1, 1)  # Reshape for sklearn compatibility

# Compute covariance matrix
cov_matrix = kernel(X)

# Plot covariance matrix
plt.figure(figsize=(8, 6))
plt.imshow(cov_matrix, cmap='viridis', origin='lower', extent=[0, 9, 0, 9])
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix for North Atlantic SST Kernel')
plt.xlabel('$x_{\mathrm{i}}$', fontsize=12)
plt.ylabel('$x_{\mathrm{j}}$', fontsize=12)
plt.savefig("NASST_COVMATRIX.png", dpi=1000)
plt.show()
