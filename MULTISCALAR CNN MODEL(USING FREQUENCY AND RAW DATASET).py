!gdown --id 1ze6D6REFbDdyAXsxN5kwEqfh_OkC8qTo
!unzip RF_Data_raw.zip
!rm RF_Data_raw.zip
!unzip RF_Data_raw.zip
!pip install tensorflow==2.8
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.figure
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
def decode(datum):
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y
def encode(datum):
    return to_categorical(datum)

cd /content/drive/MyDrive/ML
print("Loading Data ...")
Data = np.loadtxt("RF_Data_raw.csv", delimiter=",")
Data.shape
from google.colab import drive
drive.mount('/content/drive')
from scipy.fft import fft
time_domain = []
freq_domain = []
Label_1 = []
Label_2 = []
Label_3 = []

for data in Data.T:
    for i in range(0,50000,5000):
        segment = data[i:i+5000]
        time_domain.append(segment)
        freq_domain.append(abs(fft(segment)))
        Label_1.append([data[50000]])
        Label_2.append([data[50001]])
        Label_3.append([data[50002]])

time_domain = np.array(time_domain)
freq_domain = np.array(freq_domain)
x_r = time_domain
x_f = freq_domain
Label_1 = np.array(Label_1);    Label_1 = Label_1.astype(int);
Label_2 = np.array(Label_2);    Label_2 = Label_2.astype(int);
Label_3 = np.array(Label_3);    Label_3 = Label_3.astype(int);
y1 = encode(Label_1)
y2 = encode(Label_2)
y3 = encode(Label_3)
del Data, time_domain, freq_domain
from sklearn.model_selection import train_test_split
x_train_f, x_test_f, x_train_r, x_test_r, y_train, y_test = train_test_split(x_f, x_r, y3, test_size=0.2, random_state=43, shuffle=True)
print(x_train_r.shape)
print(x_train_f.shape)
print(y_train.shape)
print(x_test_r.shape)
print(x_test_f.shape)
print(y_test.shape)
del Label_1, Label_2, Label_3, x_r, x_f
# reshape input to be [samples, time steps, features = 1] as the model requires this 3D shape imput
x_train_r = np.reshape(x_train_r, (x_train_r.shape[0], x_train_r.shape[1], 1))
x_test_r = np.reshape(x_test_r, (x_test_r.shape[0], x_test_r.shape[1], 1))
x_train_f = np.reshape(x_train_f, (x_train_f.shape[0], x_train_f.shape[1], 1))
x_test_f = np.reshape(x_test_f, (x_test_f.shape[0], x_test_f.shape[1], 1))

print(x_train_r.shape)
print(x_test_r.shape)
print(x_train_f.shape)
print(x_test_f.shape)
from tensorflow.keras import backend as Ke
from tensorflow.keras.layers import concatenate, Input, GlobalMaxPooling1D, GlobalAveragePooling1D , Conv1D ,MaxPooling1D , Dropout
from tensorflow.keras.regularizers import l2
Ke.clear_session()
raw_input=Input(shape=(5000,1),name="rawInput")
x=Conv1D(32,7,padding='same', activation='relu',strides=2)(raw_input)
x=MaxPooling1D(pool_size=4,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(64,7,padding='same', activation='relu',strides=2)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(128,7,padding='same', activation='relu',strides=2)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

# x=Conv1D(256,7,padding='same', activation='relu',strides=1)(x)
# x=MaxPooling1D(pool_size=2,padding='same')(x)
# x=Dropout(rate = 0.25)(x)

x=Conv1D(256,7,padding='same', activation='relu',strides=1)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(512,7,padding='same', activation='relu',strides=1)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

# x=Conv1D(512,7,padding='same', activation='relu',strides=1)(x)
# x=MaxPooling1D(pool_size=2,padding='same')(x)
# x=Dropout(rate = 0.25)(x)

raw_output=GlobalMaxPooling1D()(x)
raw_model=keras.Model(raw_input, raw_output, name="rawModel")
raw_model.summary()


from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# ... (code for defining the model)

# Display the model architecture diagram
plot_model(raw_model, show_shapes=True)
plt.show()
Ke.clear_session()
freq_input=Input(shape=(5000,1),name="freqInput")
x=Conv1D(32,7,padding='same', activation='relu',strides=2)(freq_input)
x=MaxPooling1D(pool_size=4,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(64,7,padding='same', activation='relu',strides=2)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(128,7,padding='same', activation='relu',strides=2)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

x=Conv1D(256,7,padding='same', activation='relu',strides=1)(x)
x=MaxPooling1D(pool_size=2,padding='same')(x)
x=Dropout(rate = 0.25)(x)

# x=Conv1D(256,7,padding='same', activation='relu',strides=1)(x)
# x=MaxPooling1D(pool_size=2,padding='same')(x)
# x=Dropout(rate = 0.25)(x)

# x=Conv1D(512,7,padding='same', activation='relu',strides=1)(x)
# x=MaxPooling1D(pool_size=2,padding='same')(x)
# x=Dropout(rate = 0.25)(x)

# x=Conv1D(512,7,padding='same', activation='relu',strides=1)(x)
# x=MaxPooling1D(pool_size=2,padding='same')(x)
# x=Dropout(rate = 0.25)(x)

freq_output=GlobalMaxPooling1D()(x)
freq_model=keras.Model(freq_input, freq_output, name="freqModel")
freq_model.summary()

Ke.clear_session()
input_raw=Input(shape=(5000,1),name="raw")
raw=raw_model(input_raw)

input_freq=Input(shape=(5000,1),name="freq")
freq=freq_model(input_freq)


concat=concatenate([raw,freq])
hidden=Dropout(0.25)(concat)


hidden=Dense(128,activation='relu')(hidden)
hidden=Dropout(0.25)(hidden)

hidden=Dense(64,activation='relu')(hidden)
hidden=Dropout(0.25)(hidden)

hidden=Dense(32,activation='relu')(hidden)
hidden=Dropout(0.25)(hidden)

output=Dense(10,activation='softmax')(hidden)
net=tf.keras.Model([input_raw,input_freq], output, name="Net")

net.summary()
#keras.utils.plot_model(net, "net.png", show_shapes=True)
from tensorflow.keras.utils import plot_model
plot_model(net, "net.png", show_shapes=True)
def run_model():
    # learning_rate = 0.01 # initial learning rate
    # decay_rate = 0.1
    # opt=keras.optimizers.Adam(learning_rate=learning_rate)
    # def exp_decay(epoch):
    #     lrate = learning_rate * np.exp(-decay_rate*epoch)
    #     return lrate

    # # learning schedule callback
    # loss_history = tf.keras.callbacks.History()
    # lr_rate = tf.keras.callbacks.LearningRateScheduler(exp_decay)

    # estp = EarlyStopping(monitor='val_loss', min_delta=0.0005,patience=8 , verbose=1, mode='auto',restore_best_weights=True)
    # callbacks_list = [loss_history, lr_rate]

    opt=keras.optimizers.Adam(learning_rate=0.00001)
    net.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    estp = EarlyStopping(monitor='val_loss', min_delta=0.0005,patience=8 , verbose=1, mode='auto',restore_best_weights=True) #, callbacks = [estp]

    history = net.fit(
        {"raw":x_train_r,"freq":x_train_f},
        y_train,
        batch_size=128,
        epochs=300,
        verbose = 1,
        validation_split=0.2,
        callbacks = [estp])

    history_dict = history.history
    history_dict.keys()
    results = net.evaluate({"raw":x_test_r,"freq":x_test_f}, y_test)

#good hyper parameters


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define the optimizer with adjusted learning rate
opt = Adam(learning_rate=0.001)

# Compile the model with adjusted hyperparameters
net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Define early stopping with adjusted parameters
estp = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=8, verbose=1, mode='auto', restore_best_weights=True)

# Fit the model with updated hyperparameters
history = net.fit(
    {"raw": x_train_r, "freq": x_train_f},
    y_train,
    batch_size=64,  # Adjusted batch size
    epochs=500,  # Increased number of epochs
    verbose=1,
    validation_split=0.2,
    callbacks=[estp]
)

# Evaluate the model on the test set
results = net.evaluate({"raw": x_test_r, "freq": x_test_f}, y_test)
del x, raw_model, raw_input, raw_output, raw, output, input_freq, input_raw, net

del concat, data, freq, freq_input, freq_model, freq_output, hidden, i
# results = model.evaluate(x_test, y_test)
print ("Accuracy on test set:" , results)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
#PLOT TRAINING CURVES
# val_loss = history.history['val_loss']
# loss = history.history['loss']
# accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

# plt.rcParams['figure.figsize'] = [ 10,8]
plt.rcParams['figure.figsize'] = [10, 5]
ax=plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'b', label='Training loss', color='red')
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
plt.plot(epochs,val_loss , 'b', label='Validation loss', color='green')
plt.title('Training and validation loss',fontsize=15)
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()

ax=plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'b', label='Training acc', color='red')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc', color='green')
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.title('Training and validation accuracy', fontsize=15)
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)
plt.legend()
plt.tight_layout()
plt.show()
q=np.array(Y_pred) #4-class
label_l=[]
for i in range (q.shape[0]):
  if q[i]==0:
    label_l.append('No Drone')
  elif q[i]==1:
    label_l.append('Bebop')
  elif q[i]==2:
    label_l.append('AR')
  elif q[i]==3:
    label_l.append('Phantom')

Label_ar=np.array(label_l)
q=np.array(Y_pred) #10-class ','Bebop m2', 'Bebop m3', 'Bebop m4', 'AR m1', 'AR m2', 'AR m3', 'AR m4', 'Phantom m1
label_l=[]
for i in range (q.shape[0]):
  if q[i]==0:
    label_l.append('No Drone')
  elif q[i]==1:
    label_l.append('Bebop m1')
  elif q[i]==2:
    label_l.append('Bebop m2')
  elif q[i]==3:
    label_l.append('Bebop m3')
  elif q[i]==4:
    label_l.append('Bebop m4')
  elif q[i]==5:
    label_l.append('AR m1')
  elif q[i]==6:
    label_l.append('AR m2')
  elif q[i]==7:
    label_l.append('AR m3')
  elif q[i]==8:
    label_l.append('AR m4')
  elif q[i]==9:
    label_l.append('Phantom m1')

Label_ar=np.array(label_l)
#TSNE PLOTS
import seaborn as sns
from sklearn.manifold import TSNE
# out_dense= keras.Model(inputs=model.input,outputs=model.get_layer('dense_5').output)
# Y_denseout = out_dense.predict(x_test) #predicted Y

tsne = TSNE(n_components=2,learning_rate='auto',init='random')
Y_embedded = tsne.fit_transform(predicted)
plt.figure(figsize=(10,8))
sns.scatterplot(Y_embedded[:,0], Y_embedded[:,1], hue=Label_ar)
plt.xlabel("Dimension 1",fontweight ='bold', fontsize = 15);
plt.ylabel("Dimenssion 2",fontweight ='bold', fontsize = 15)
plt.title('t-SNE plot After applying Lightweight CNN',fontweight ='bold', fontsize = 15)
