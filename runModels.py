import numpy as np
from vgg import CreateVGG16Model, CreateSimpleFCN, AddCRFRNNToModel
from dataset_utils import getStanfordImagesAndLabels, getWBCImagesAndLabels
from keras import optimizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model


STANFORD_PARAMS = {
    'num_classes': 8,
    'input_shape': (240, 320, 3)
}

WBC_PARAMS = {
    'num_classes': 3,
    'input_shape': (120, 120, 3)
}

def getModelPredictions(modelFile, datatype, outputDir, batch_size, model=None):
    if model is None:
        model = load_model(modelFile)
    if datatype == "Stanford":
        X_test, y_test = getStanfordImagesAndLabels("test")
    else:
        X_test, y_test = getWBCImagesAndLabels("test")
    predictions = model.predict(X_test)
    print("done predicting")
    pickle.dump(predictions, open("%s/predictions.pkl"%  outputDir, "wb"))

def runCRFCNNModel(modelFile, dataType="Stanford", lr=0.001, epochs=10,
                   plotPrefix="", data=None,freezeUnderlyingModel=False):
    model = load_model(modelFile)
    if data is not None:
        X_train, y_train = data
    if dataType == "Stanford":
        num_classes = STANFORD_PARAMS['num_classes']
        input_shape = STANFORD_PARAMS['input_shape']
        if data is None:
            X_train, y_train = getStanfordImagesAndLabels("training")
    else:
        num_classes = WBC_PARAMS['num_classes']
        input_shape = WBC_PARAMS['input_shape']
        if data is None:
            X_train, y_train = getWBCImagesAndLabels("training")
    dims = (input_shape[0], input_shape[1])
    model = AddCRFRNNToModel(model, dims, num_classes, freezeUnderlyingModel, "Cropping")
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=1, validation_split=0.2)
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.savefig("%sloss.png" % plotPrefix)
    plt.clf()
    plt.plot(history.history['acc'], label="acc")
    plt.plot(history.history['val_acc'], label="val acc")
    plt.savefig("%sacc.png" % plotPrefix)
    pickle.dump(history.history, open("%shistory.pkl"% plotPrefix, "wb"))
    model.save("%smodel.h5" % plotPrefix)
    return history.history["val_acc"][-1], model
    

def runModel(dataType="Stanford", modeltype="VCG", lr=0.001, batch=5, epochs=10, regParam=0.001, plotPrefix="", data=None):
    if data is not None:
        X_train, y_train = data
    if dataType == "Stanford":
        num_classes = STANFORD_PARAMS['num_classes']
        input_shape = STANFORD_PARAMS['input_shape']
        if data is None:
            X_train, y_train = getStanfordImagesAndLabels("training")
    else:
        num_classes = WBC_PARAMS['num_classes']
        input_shape = WBC_PARAMS['input_shape']
        if data is None:
            X_train, y_train = getWBCImagesAndLabels("training")
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
    pickle.dump(history.history, open("%shistory.pkl"% plotPrefix, "wb"))
    model.save("%smodel.h5" % plotPrefix)
    return history.history["val_acc"][-1], model

def runCrossValidation(modelName="VCG", lr=[0.001], regs=[0.001], batch=10, epochs=10):
    results = []
    data = getStanfordImagesAndLabels("training")
    for l in lr:
        for r in regs:
            prefix = "l%s_r%s" % (l, r)
            val_acc, _ = runModel(modelName, lr=l, regParam=r, batch=batch, plotPrefix=prefix, epochs=epochs, data=data)
            results.append((l,r,val_acc))
            print("%s, %s, %s" % (l, r, val_acc))

# runModel(dataType="WBC", modeltype="VCG", lr=0.001, batch=5, epochs=50, regParam=0.001, plotPrefix="vggWBCResult/", data=None)
# runModel(dataType="WBC", modeltype="Simple", lr=0.001, batch=5, epochs=50, regParam=0.005, plotPrefix="fcnWBCResult/", data=None)

_, modelResult = runCRFCNNModel("vggWBCResult/model.h5", dataType="WBC", lr=0.005, epochs=10, plotPrefix="crfVGGFreezeWBC/", data=None,freezeUnderlyingModel=True)
getModelPredictions("crfVGGFreezeWBC/model.h5", "WBC", "results/crfFreezeWBC", 1, modelResult)

# Cross Validation
# runCrossValidation("VCG", lr=[0.01, 0.001, 0.005], regs=[0.01, 0.001, 0.0001], epochs=5)
# runCrossValidation("Simple", lr=[0.1, 0.01, 0.001, 0.005], regs=[0.01], epochs=5)