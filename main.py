import mnistdata
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from tensorflow import set_random_seed

set_random_seed(1)
np.random.seed(1)

def main():
    mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

    model = simple_model()

    print("\n|---- TRAINING ----|")

    batch_size=100

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(flatten_images(mnist.train.images), mnist.train.labels, epochs=10, batch_size=batch_size)

    print("\n|---- EVALUATING ----|")

    score = model.evaluate(flatten_images(mnist.test.images), mnist.test.labels, batch_size=batch_size)

    print(model.metrics_names[1] + ": ", "%.2f" % (score[1]*100), "%")

    return

def simple_model():
    model = Sequential()

    model.add(Dense(30))
    model.add(Dense(10, activation='softmax'))

    return model

def more_layers():
    model = Sequential()
    
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(60, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    return model

# Data transformers

def flatten_images(arr):
    return np.reshape(arr, (arr.shape[0], arr.shape[1]*arr.shape[2]))

if __name__ == "__main__":
    main()