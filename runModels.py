import numpy as np
from vgg import CreateVGG16Model
from dataset_utils import getImagesAndLabels
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_shape = (240, 320, 3)
label_input = (240, 320)
num_classes = 8
X_train, y_train = getImagesAndLabels("training")
# model = CreateVGG16Model(input_shape, num_classes)
model = CreateVGG16Model(input_shape, num_classes)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print(len(y_train), y_train.shape)
history = model.fit(X_train, y_train, epochs=10, batch_size=2)
plt.plot(history.history['loss'])
plt.savefig("history.png")