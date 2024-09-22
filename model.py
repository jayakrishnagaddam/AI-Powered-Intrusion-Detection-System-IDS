import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

file_path = 'CIC Dataset\Friday-WorkingHours-Morning.pcap_ISCX.csv'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

df_clean = df.drop(['Flow ID', 'Timestamp'], axis=1, errors='ignore')
label_encoder = LabelEncoder()
df_clean['Label'] = label_encoder.fit_transform(df_clean['Label'])
df_clean.fillna(0, inplace=True)

df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.fillna(0, inplace=True)

X = df_clean.drop('Label', axis=1)
y = df_clean['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_samples = X_scaled.shape[0]
num_features = X_scaled.shape[1]

height = 6
width = 13

if num_features != height * width:
    raise ValueError("The number of features does not match the specified height and width.")

X_reshaped = X_scaled.reshape(num_samples, height, width, 1)

y_categorical = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(height, width, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

predictions = model.predict(X_test)
