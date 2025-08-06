%cd /content/drive/MyDrive/ML
!unzip RF_Data_raw.zip
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


chunk_size = 5000  
df_chunks = pd.read_csv('/content/drive/MyDrive/ML/RF_Data_raw.csv', chunksize=chunk_size)


X_data = []
y_data = []


for chunk in df_chunks:
    
    X_chunk = chunk.iloc[:, 1:].values 
    y_chunk = chunk.iloc[:, 0].values   

   
    X_data.append(X_chunk)
    y_data.append(y_chunk)


X_data = np.concatenate(X_data, axis=0)
y_data = np.concatenate(y_data, axis=0)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1], 1)
X_train = X_train.reshape(-1, *input_shape)
X_test = X_test.reshape(-1, *input_shape)


model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stop = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/nandu/best_model.h5', save_best_only=True)

steps_per_epoch = len(X_train) // batch_size
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch, callbacks=[early_stop, model_checkpoint])


model.load_weights('/content/drive/MyDrive/nandu/best_model.h5')


loss, accuracy = model.evaluate(X_test, y_test)
print("Testing Loss:", loss)
print("Testing Accuracy:", accuracy)
