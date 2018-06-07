import numpy as np
from vgg import CreateVGG16Model, CreateSimpleFCN
from dataset_utils import getImagesAndLabels
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STANFORD_PARAMS = {
    'num_classes': 8,
    'input_shape': (240, 320, 3)
}

def runModel(modeltype="VCG", lr=0.001, batch=5, epochs=10, regParam=0.001, plotPrefix="", data=None):
    num_classes = STANFORD_PARAMS['num_classes']
    input_shape = STANFORD_PARAMS['input_shape']
    if data is None:
        X_train, y_train = getImagesAndLabels("training")
    else:
        X_train, y_train = data
    # model = CreateVGG16Model(input_shape, num_classes)
    if modeltype == "VCG":
        model = CreateVGG16Model(input_shape, num_classes, regParam=regParam)
    else:
        model = CreateSimpleFCN(input_shape, num_classes, regParam=regParam)
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print(len(y_train), y_train.shape)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=0.2)
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.savefig("%sloss.png" % plotPrefix)
    plt.clf()
    plt.plot(history.history['acc'], label="acc")
    plt.plot(history.history['val_acc'], label="val acc")
    plt.savefig("%sacc.png" % plotPrefix)
    return history.history["val_acc"][-1]

def runCrossValidation(modelName="VCG", lr=[0.001], regs=[0.001], batch=10, epochs=10):
    results = []
    data = getImagesAndLabels("training")
    for l in lr:
        for r in regs:
            prefix = "l%s_r%s" % (l, r)
            val_acc = runModel(modelName, lr=l, regParam=r, batch=batch, plotPrefix=prefix, epochs=epochs, data=data)
            results.append((l,r,val_acc))
            print("%s, %s, %s" % (l, r, val_acc))

# runCrossValidation("VCG", lr=[0.01, 0.001, 0.005], regs=[0.01, 0.001, 0.0001], epochs=5)
runCrossValidation("Simple", lr=[0.1, 0.01, 0.001, 0.005], regs=[0.01], epochs=5)