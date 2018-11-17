import mnistdata
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)



model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

batchsize = 100

flattened_images = np.reshape(mnist.train.images, (mnist.train.images.shape[0], 28*28))

model.fit(flattened_images, mnist.train.labels, epochs=1, batch_size=100)

