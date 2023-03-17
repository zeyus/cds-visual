# train a simple tensorflow model on cifar-10 dataset
# https://www.cs.toronto.edu/~kriz/cifar.html

import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
# load cifar-10 dataset
from tensorflow.keras.datasets import cifar10

# file directory
file_dir = Path(__file__).parent

# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(3000, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
n_epochs = 10
# train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epochs, batch_size=128)

# evaluate model
model.evaluate(x_test, y_test)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
predictions = model.predict(x_test)
report = classification_report(y_test, np.argmax(predictions, axis=1), target_names=classes)
# save report
with open(file_dir / 'cifar10_model_report.txt', 'w') as f:
    f.write(report)

# save model
model.save(file_dir / 'cifar10_model.h5')

# save model statistics and plots
# save model summary
with open(file_dir / 'cifar10_model_summary.txt', 'w') as f:
    model.summary(
        print_fn=lambda x: f.write(x + '\n')
    )

# save model history
with open(file_dir / 'cifar10_model_history.txt', 'w') as f:
    f.write(str(model.history.history))
print(history.history.keys())
# save model accuracy plot
plt.style.use("fivethirtyeight")
plt.figure()
plt.plot(np.arange(0, n_epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, n_epochs), history.history["val_loss"], label="val_loss", linestyle=":")
plt.plot(np.arange(0, n_epochs), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n_epochs), history.history["val_accuracy"], label="val_acc", linestyle=":")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.tight_layout()
plt.legend()
plt.savefig(file_dir / 'cifar10_model_history.png')

