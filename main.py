import mnistdata
from keras.models import Sequential
import keras.layers as layers
import numpy as np
from tensorflow import set_random_seed
import argparse
from keras.utils import plot_model

set_random_seed(1)
np.random.seed(1)

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate DigitNet to predict handwritten digits')
    parser.add_argument('--epochs', default=5, type=int,
                help='The number of epochs to run the model')
    parser.add_argument('--model', default='multilayer', 
                help='Specifies which model to run, available models are: simple, sigmoid, multilayer, convolutional')

    args = parser.parse_args()

    if args.model == 'simple':
        model, data_transformer = simple()
    elif args.model == 'sigmoid':
        model, data_transformer = sigmoid()
    elif args.model == 'multilayer':
        model, data_transformer = multilayer()
    elif args.model == 'convolutional':
        model, data_transformer = convolutional()

    print('Summary of model:')
    print(model.summary())

    mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

    print("\n|---- TRAINING ----|")

    batch_size=100

    training_images = data_transformer(mnist.train.images)
    testing_images = data_transformer(mnist.test.images)
    
    print(np.shape(training_images))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, mnist.train.labels, epochs=args.epochs, batch_size=batch_size)

    print("\n|---- EVALUATING ----|")

    score = model.evaluate(testing_images, mnist.test.labels, batch_size=batch_size)

    print("Accuracy: ", "%.2f" % (score[1]*100), "%")

    return

def simple():
    model = Sequential()

    model.add(layers.Dense(30))
    model.add(layers.Dense(10, activation='softmax'))

    return model, flatten_images

def sigmoid():
    model = Sequential()

    model.add(layers.Dense(30, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    return model, flatten_images

def multilayer():
    model = Sequential()
    
    model.add(layers.Dense(200, activation='sigmoid'))
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(layers.Dense(60, activation='sigmoid'))
    model.add(layers.Dense(30, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))

    return model, flatten_images

def convolutional():
    model = Sequential()

    model.add(layers.Conv2D(
        input_shape=(28, 28, 1),
        padding='same',
        filters=6,
        kernel_size=6,
        strides=1
    ))

    model.add(layers.Conv2D(
        filters=12,
        padding='same',
        kernel_size=5,
        strides=2
    ))

    model.add(layers.Conv2D(
        filters=24,
        padding='same',
        kernel_size=4,
        strides=2
    ))

    model.add(layers.Flatten())

    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dropout(0.75))
    model.add(layers.Dense(10, activation='softmax'))


    return model, empty_transformer

def flatten_images(arr):
    return np.reshape(arr, (arr.shape[0], arr.shape[1]*arr.shape[2]))

def empty_transformer(arr):
    return arr

if __name__ == "__main__":
    main()