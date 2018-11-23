# 1.加载数据集
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)
print(train_data[0])
# 解码评论数据
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# 2.将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 训练测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(train_data[0])
# 训练测试标签向量化
y_train = np.asarray(train_labels, dtype=np.float32)
y_test = np.asarray(test_labels, dtype=np.float32)

# 3.构建网络
net = models.Sequential()
net.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
net.add(layers.Dense(16, activation="relu"))
net.add(layers.Dense(1, activation="sigmoid"))

# 4.编译模型
net.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=['accuracy']
)
#
# # 5.配置优化器
# net.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss="binary_crossentropy",
#     metrics=['accuracy']
# )
#
# # 6.自定义损失函数和指标
# net.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss=losses.binary_crossentropy,
#     metrics=[metrics.binary_accuracy]
# )

# 7.留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 8.训练模型
history = net.fit(
    partial_x_train,
    partial_y_train,
    batch_size=512,
    epochs=20,
    validation_data=(x_val, y_val)
)

history_dict = history.history
print(history_dict.keys())

# 9.绘制训练损失和验证损失
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 10.绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 11.重新训练一个4批次网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print()
print(results)
print()
print(model.predict(x_test))
