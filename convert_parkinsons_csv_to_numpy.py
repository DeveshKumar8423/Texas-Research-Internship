import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("parkinsons.data")

# Drop name column
df = df.drop(columns=["name"])

# Split features and target
X = df.drop(columns=["status"]).values
y = df["status"].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape for time-series format: [samples, time_steps]
# Here we treat each feature vector as a "sequence" of length n_features
X = X.reshape((X.shape[0], X.shape[1]))

# Create dummy Y (required for Motion Code, just repeat X here)
Y = X.copy()

# Split
X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(X, Y, y, test_size=0.2, random_state=42)

# Save
train_dict = {"X": X_train, "Y": Y_train, "labels": y_train}
test_dict = {"X": X_test, "Y": Y_test, "labels": y_test}
np.save("data/parkinson_pd1/train.npy", train_dict)
np.save("data/parkinson_pd1/test.npy", test_dict)

print("Saved train.npy and test.npy in dictionary format.")
