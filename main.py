import mnistdata
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from tensorflow import set_random_seed

set_random_seed(1)
np.random.seed(1)

def main():
    mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

    model, data_transformer = simple_model()

    print("\n|---- TRAINING ----|")

    batch_size=100

    model.fit(data_transformer(mnist.train.images), mnist.train.labels, epochs=1, batch_size=batch_size)

    print("\n|---- EVALUATING ----|")

    score = model.evaluate(data_transformer(mnist.test.images), mnist.test.labels, batch_size=batch_size)

    print(model.metrics_names[1] + ": ", "%.2f" % (score[1]*100), "%")

    return

def simple_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, flatten_images

# Data transformers

def flatten_images(arr):
    return np.reshape(arr, (arr.shape[0], arr.shape[1]*arr.shape[2]))

def emptpy_transformer(arr):
    return arr

if __name__ == "__main__":
    main()