import numpy as np
from keras import layers
from keras import models
from keras.layers import Dense
from keras.utils import to_categorical,vis_utils

import matplotlib.pyplot as plt 



def load_data(path='mnist.npz'):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('mnist.npz')

## 数据摊平，归一化
x_train = x_train.reshape(60000,28*28)
x_train = x_train.astype('float32')/255

x_test = x_test.reshape(10000,28*28)
x_test = x_test.astype('float32')/255

#准备标签
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def main():
    print(x_train.shape,y_train)
    print(x_test.shape,y_test)
    model = models.Sequential()
    model.add(Dense(512,activation='relu',input_shape=(28*28,)))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit(x_train,y_train, validation_split=0.25,epochs=5,batch_size=128,verbose=1)
    test_loss,test_acc = model.evaluate(x_test,y_test)
    print("test_loss: {},test_acc: {}".format(test_loss,test_acc))

    print(history.history)
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()



