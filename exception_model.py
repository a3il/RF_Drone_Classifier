import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
from google.colab import drive

drive.mount('/content/drive')

spectrogram_path = '/content/drive/MyDrive/Adi/Spectrograms'
label_path = '/content/drive/MyDrive/Adi/file_labels.csv'

learning_rate = 0.001
dropout_rate = 0.5
epochs = 15
batch_size = 64

spectrograms = []
labels = []

label_df = pd.read_csv(label_path)

for index, row in label_df.iterrows():
    filename = row['File_Name']
    label = row['Label']
    filename = os.path.splitext(filename)[0] + '.png'

    image_path = os.path.join(spectrogram_path, filename)

    spectrogram = Image.open(image_path).convert('RGB')
    spectrogram = spectrogram.resize((496, 369))
    spectrograms.append(np.array(spectrogram))
    labels.append(label)

spectrograms = np.array(spectrograms)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    spectrograms, labels_categorical, test_size=0.2, random_state=42
)

base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(369, 496, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax', kernel_initializer=glorot_uniform()))

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('/content/drive/MyDrive/Adi/training_curvess1.png')
plt.show()

X_test_features = base_model.predict(X_test)

num_samples, height, width, channels = X_test_features.shape
X_test_reshaped = np.reshape(X_test_features, (num_samples, height * width * channels))

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_test_reshaped)

plt.figure(figsize=(8, 6))
for class_index in range(num_classes):
    indices = np.where(y_test.argmax(axis=1) == class_index)
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], label=class_names[class_index])
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.legend()
plt.savefig('/content/drive/MyDrive/Adi/tsne_embeddingss1.png')
plt.show()

model.save('/content/drive/MyDrive/Adi/mobilenet_model1.h5')
