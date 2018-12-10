import os
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 6
batch_size = 64
nb_epoch = 50

PROVINCES = ("京" ,"闽" ,"粤" ,"苏" ,"沪" ,"浙")
nProvinceIndex = 0

input_count = 0
for i in range(0, NUM_CLASSES):
    dir = './tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1

# 定义对应维数和各维长度的数组
input_images = np.array([[0] * SIZE for i in range(input_count)])
input_labels = np.array([[0] * NUM_CLASSES for i in range(input_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(0, NUM_CLASSES):
    dir = './tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = Image.open(filename)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 190:
                        input_images[index][w + h * width] = 0
                    else:
                        input_images[index][w + h * width] = 1
            input_labels[index][i] = 1
            index += 1

# 第一次遍历图片目录是为了获取图片总数
val_count = 0
for i in range(0, NUM_CLASSES):
    dir = './tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            val_count += 1

# 定义对应维数和各维长度的数组
val_images = np.array([[0] * SIZE for i in range(val_count)])
val_labels = np.array([[0] * NUM_CLASSES for i in range(val_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(0, NUM_CLASSES):
    dir = './tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = Image.open(filename)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        val_images[index][w + h * width] = 0
                    else:
                        val_images[index][w + h * width] = 1
            val_labels[index][i] = 1
            index += 1
print(input_images.shape)
print(input_labels.shape)

X_train = input_images.astype('float32')
X_test = val_images.astype('float32')
# 归一化到0-1
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to_categorical(y, nb_classes=None)
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
# y: 类别向量; nb_classes:总共类别数
#Y_train = np_utils.to_categorical(input_labels, NUM_CLASSES)
#Y_test = np_utils.to_categorical(val_labels, NUM_CLASSES)
Y_train = input_labels
Y_test = val_labels
# Dense层:即全连接层
# keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)


model = Sequential()
model.add(Dense(512, input_shape=(1280,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递activation参数实现。
# 以下两行等价于：model.add(Dense(512,activation='relu'))
model.add(Dense(512))
model.add(Activation('relu'))

# Dropout  需要断开的连接的比例
model.add(Dropout(0.2))
model.add(Dense(6))
model.add(Activation('softmax'))

# 打印出模型概况
#print('model.summary:')
#model.summary()

# 在训练模型之前，通过compile来对学习过程进行配置
# 编译模型以供训练
# 包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']
# 如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练模型
# Keras以Numpy数组作为输入数据和标签的数据类型
# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
# nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
History = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=2, validation_data=(X_test, Y_test))


# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# 按batch计算在某些输入数据上模型的误差
print('-------evaluate--------')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('.\chinese_character_model.h5')