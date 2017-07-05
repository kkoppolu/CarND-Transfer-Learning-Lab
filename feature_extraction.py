import pickle
import tensorflow as tf
from keras.layers.core import Activation, Dense, Flatten
from keras.layers import Input
from keras.models import Sequential
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    nb_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # TODO: train your model here
    model.fit(X_train, y_train, nb_epoch=50, batch_size=256)
    metrics = model.evaluate(X_val, y_val, batch_size=32)

    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]

    print('{}: {}'.format(metric_name, metric_value))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
