# 1.加载数据集
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(train_labels)
print(test_images.shape)
print(test_labels.shape)
print(test_labels)

# 2.构建网络
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# 3.编译网络
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4.数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 5.准备标签
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 6.训练以及评估
network.fit(train_images, train_labels, batch_size=128, epochs=5)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss, test_acc)
