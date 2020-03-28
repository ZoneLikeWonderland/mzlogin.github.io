---
layout: post
title: 入门Keras记录 
categories: [learning]
description: MNIST手写数字识别 
keywords: Keras, 深度学习
---

## 参考
- [tensorflow-101教程](https://github.com/geektime-geekbang/tensorflow-101)
- [Keras文档](https://keras.io/zh/models/sequential/)



## 配置TensorFlow和Keras的GPU显存
```python
# must do at first
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
```



## 使用Sequential顺序模型

[Sequential API](https://keras.io/zh/models/sequential/)

### 定义CNN网络

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
## Feature Extraction
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape)) # input_shape=(28,28,1)

# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化层，池化窗口 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

## Classification
# 全联接层
model.add(Dense(128, activation='relu'))

# Dropout 50% 的输入神经元
model.add(Dropout(0.5))

# 使用 softmax 激活函数做多分类，输出各数字的概率
model.add(Dense(n_classes, activation='softmax'))
```

使用`model.summary()`查看模型结构：
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1179776   
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________
```



### 配置学习过程

[model.compile()](https://keras.io/zh/models/sequential/#compile)

```python
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], 
              optimizer='adam')
```



### 训练模型

[model.fit()](https://keras.io/zh/models/sequential/#fit)

```python
history = model.fit(X_train, # X_train.shape==(None,28,28,1)
                    Y_train, # Y_train.shape==(None,10)
                    batch_size=128,
                    epochs=5,
                    verbose=2,
                    validation_data=(X_test, Y_test))
```

使用`history.history`查看模型训练损失和评估值：

```python
{'val_loss': [0.052851562236621977,
  0.040380205494537948,
  0.032175659690238535,
  0.02983101295363158,
  0.026629098207131028],
 'val_acc': [0.98260000000000003,
  0.9869,
  0.98899999999999999,
  0.99019999999999997,
  0.9909],
 'loss': [0.24002769051790238,
  0.083018691488107046,
  0.061485205243031187,
  0.051883092286189397,
  0.043534082640707496],
 'acc': [0.92668333336512243,
  0.97501666669845577,
  0.98131666666666661,
  0.98446666663487747,
  0.98626666666666662]}
```



### 保存模型

[model.save()](https://keras.io/zh/getting-started/faq/#how-can-i-save-a-keras-model)

```python
model.save(model_path)
```



### 加载模型

```python
from keras.models import load_model
mnist_model = load_model(model_path)
```



### 统计测试集预测结果

```python
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
    
print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1]*100))

predicted_classes = mnist_model.predict_classes(X_test)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count: {}".format(len(correct_indices)))
print("Classified incorrectly count: {}".format(len(incorrect_indices)))

```



### 预测

```python
model.predict(X_test[0:1])
```

```python
array([[  3.40244543e-12,   1.17982679e-10,   2.55023846e-09,
          1.81103132e-08,   5.43577553e-12,   5.70357997e-12,
          5.29334235e-16,   9.99999881e-01,   4.41497637e-11,
          1.44942419e-07]], dtype=float32)
```

