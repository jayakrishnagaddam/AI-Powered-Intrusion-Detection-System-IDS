import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = 'CIC Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Clean the data
df_clean = df.drop(['Flow ID', 'Timestamp'], axis=1, errors='ignore')
label_encoder = LabelEncoder()
df_clean['Label'] = label_encoder.fit_transform(df_clean['Label'])
df_clean.fillna(0, inplace=True)

# Replace inf values with 0
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.fillna(0, inplace=True)

# Split features and labels
X = df_clean.drop('Label', axis=1)
y = df_clean['Label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the new shape for Conv2D input
num_samples = X_scaled.shape[0]
num_features = X_scaled.shape[1]

# Choose appropriate height and width for the input
height = 6  # Number of rows
width = 13  # Number of columns

# Check if the features fit the specified height and width
if num_features != height * width:
    raise ValueError("The number of features does not match the specified height and width.")

# Reshape data for Conv2D
X_reshaped = X_scaled.reshape(num_samples, height, width, 1)

# Convert labels to categorical
y_categorical = to_categorical(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Build the Conv2D model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduces size to (3, 6, 32)

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduces size to (1, 3, 64)

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer for binary classification (benign/malicious)
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Predictions
predictions = model.predict(X_test)
