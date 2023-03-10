import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from PIL import Image
import numpy as np

# 設定圖片的寬、高和通道數
img_width, img_height = 28, 28
num_channels = 1

# 設定類別數量
num_classes = 10

# 設定超參數
batch_size = 128
num_epochs = 10

# 載入MNIST數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 根據Keras的要求調整數據格式
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], num_channels, img_width, img_height)
    x_test = x_test.reshape(x_test.shape[0], num_channels, img_width, img_height)
    input_shape = (num_channels, img_width, img_height)
else:
    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_width, img_height, num_channels)
    input_shape = (img_width, img_height, num_channels)

# 將像素值縮放到[0, 1]的範圍內
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 將標籤轉換為one-hot向量
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 建立模型
model = Sequential()

# 第一層卷積層，32個3x3的卷積核，使用ReLU激活函數
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# 第二層卷積層，64個3x3的卷積核，使用ReLU激活函數
model.add(Conv2D(64, (3, 3), activation='relu'))

# 第一層池化層，使用MaxPooling，將特徵圖大小減半
model.add(MaxPooling2D(pool_size=(2, 2)))

# 以一定概率進行Dropout，避免過擬合
model.add(Dropout(0.25))

# 將特徵圖展開成向量形式，準備進入全連接層
model.add(Flatten())

# 第一層全連接層，128個神經元，使用ReLU激活函數
model.add(Dense(128, activation='relu'))

# 以一定概率進行Dropout，避免過擬合
model.add(Dropout(0.5))

# 第二層全連接層，輸出維度為類別數量，使用Softmax激活函數
model.add(Dense(num_classes, activation='softmax'))

# 編譯模型，指定損失函數、優化器和評估指標
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 將訓練集拆分為訓練集和驗證集
split_idx = int(len(x_train) * 0.8)
x_train, x_val = x_train[:split_idx], x_train[split_idx:]
y_train, y_val = y_train[:split_idx], y_train[split_idx:]

# 訓練模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val))

# 評估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 讀取待預測圖片，並轉換為灰度圖
img = Image.open(r'*').convert('L')

# 調整圖片大小為(28, 28)
img = img.resize((28, 28))

# 將圖片轉換為Numpy數組
x = np.array(img)

# 將像素值縮放到[0, 1]的範圍內
x = x.astype('float32') / 255.

# 根據Keras的要求調整數據格式
if K.image_data_format() == 'channels_first':
    x = x.reshape(1, 1, img_width, img_height)
else:
    x = x.reshape(1, img_width, img_height, 1)

# 使用模型進行預測
y_pred = model.predict(x)

# 獲取預測結果
predicted_class = np.argmax(y_pred)

# 印出預測結果
print('Predicted class:', predicted_class)