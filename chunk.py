import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

seq_length = 50000
num_ch = 1
filters1 = 256
filters2 = 128
filters3 = 64
kernel_size = 5
dropout_rate = 0.25
dense_units1 = 64
dense_units2 = 32
num_classes = 4
learning_rate = 0.001
batch_size = 32
epochs = 100

model = Sequential()
model.add(Dense(filters3, kernel_size, padding='valid', activation='relu', strides=2, input_shape=(seq_length, num_ch)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=dropout_rate))
model.add(Flatten())

model.add(Dense(dense_units1, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(dense_units2, activation='relu'))
model.add(Dropout(rate=dropout_rate))

model.add(Dense(num_classes, activation='softmax'))

optimizer = keras.optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

results = model.evaluate(x_test, y_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test_classes, y_pred_classes)
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Performance Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('performance_matrix.png')
plt.show()

val_loss = history.history['val_loss']
loss = history.history['loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training Loss', color='red')
plt.plot(epochs, val_loss, 'b', label='Validation Loss', color='green')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy', color='red')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy', color='green')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
