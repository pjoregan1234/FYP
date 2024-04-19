import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import datetime

data = "CLOUD_OPTICAL_THICKNESS"

df = pd.read_csv(data + "_berlin_timeseries_1pw.csv")
print(df)

threshold = -0.0001
df = df[df["default_" + data + "_mean"] >= threshold]

# Convert 'Date' column to datetime format
df['interval_from'] = pd.to_datetime(df['interval_from'])

# Calculate decimal fraction of the year
df['Decimal Year'] = df['interval_from'].dt.year + df['interval_from'].dt.dayofyear / \
    (365 if pd.to_datetime(df['interval_from'].dt.year, format='%Y').dt.is_leap_year.all() else 366)

print(df["Decimal Year"])

plt.plot(df["Decimal Year"], df["default_" + data + "_mean"], color="blue")
plt.ylabel(data + " Concentration (mol/m^2)")
plt.xlabel("Year")
plt.title("North Atlantic's " + data + " Concentration vs Time")
plt.xlim(df["Decimal Year"].min(), df["Decimal Year"].max())
plt.tight_layout()
plt.show()

X = df["Decimal Year"].to_numpy().reshape(-1, 1)
y = df["default_" + data + "_mean"].to_numpy()
#y = y*(1e3)
# Split value
split_value = 2023
# Boolean indexing to split the array
X1 = X[X < split_value].reshape(-1,1)
X2 = X[X >= split_value].reshape(-1,1)
y1 = y[:len(X1)]
y2 = y[len(X1):]


rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y1.size), size=150, replace=False)
X_train, y_train = X1[training_indices], y1[training_indices]

plt.scatter(X_train, y_train)
plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF
long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0, length_scale_bounds=(1e-20, 1e20))

from sklearn.gaussian_process.kernels import ExpSineSquared
seasonal_kernel = (2**2 * RBF(length_scale=100.0, length_scale_bounds=(1e-15, 1e15))
    * ExpSineSquared(length_scale=10.0, length_scale_bounds=(1e-3, 1e3), periodicity=1.0, periodicity_bounds="fixed"))

from sklearn.gaussian_process.kernels import WhiteKernel
noise_kernel = WhiteKernel(
   noise_level=0.1**2, noise_level_bounds=(1e-10,1e10))

# Thus, our final kernel is an addition of all previous kernel.
temp_kernel = (long_term_trend_kernel + seasonal_kernel
                + noise_kernel)

gaussian_process = GaussianProcessRegressor(kernel=temp_kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
print(gaussian_process.kernel_)

today = datetime.datetime.now()
current_month = today.year + today.month / 12
X_test = np.linspace(start=2018.4, stop=2025, num = 1_000).reshape(-1, 1)
mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)


plt.plot(X1, y1, color="black", label="Measurements (Trained on)")
plt.plot(X2, y2, color="red", label="Measurements (Not Trained on)")
plt.plot(X_test, mean_prediction, color="tab:blue", alpha=0.4, label="Gaussian Process (95% confidence interval)")
plt.fill_between(
    X_test.ravel(),
    mean_prediction - 1.96*std_prediction,
    mean_prediction + 1.96*std_prediction,
    color="tab:blue",
    alpha=0.2,
)
plt.xlim(2018.4, 2025)
plt.legend(fontsize="small")
plt.xlabel("Year")
plt.ylabel("Cloud Optical Thickness (m)")
plt.title("Berlin's Cloud Optical Thickness vs Time")
#plt.savefig("Corks_CLOUD_OPTICAL_THICKNESS_GP.png")
plt.tight_layout()
plt.show()

# Define input space
N = 100
Z = np.linspace(0, 9, N).reshape(-1, 1)  # Reshape for sklearn compatibility
# Compute covariance matrix
cov_matrix = gaussian_process.kernel_(Z)
# Plot covariance matrix
plt.figure(figsize=(8, 6))
plt.imshow(cov_matrix, cmap='viridis', origin='lower', extent=[0, 9, 0, 9])
plt.colorbar(label='Covariance')
plt.title("Covariance Matrix for Berlin's Cloud Optical Thickness Kernel")
plt.xlabel('$x_{\mathrm{i}}$', fontsize=12)
plt.ylabel('$x_{\mathrm{j}}$', fontsize=12)
#plt.savefig("Corks_CLOUD_OPTICAL_THICKNESS_COV.png")
plt.show()