import tensorflow as tf
from imageProcessing import imageProcessing
IMAGE_SHAPE = (160, 160,3)
baseModel = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
										  include_top=False, 
										  weights='imagenet')
# baseModel.trainable = False
BATCH_SIZE = 10
dataObj = imageProcessing()
dataSetTrain = dataObj.buildData("train")

# dataSetTrain = dataSetTrain.shuffle(buffer_size = dataObj.numExp)
dataSetTrain = dataSetTrain.batch(BATCH_SIZE)
# print dataSetTrain
# dataSetTest = dataObj.buildData("test")

# dataSetTest = dataSetTest.shuffle(buffer_size = dataObj.numExp)
# dataSetTest = dataSetTest.batch(BATCH_SIZE)
# print baseModel.summary()
# print imgBat
steps_per_epoch = (int(round(112+52)/BATCH_SIZE)+1)*2
glob2D = tf.keras.layers.GlobalAveragePooling2D()
predLay = tf.keras.layers.Dense(1)
# printNode = tf.Print(predLay, [tf.shape(predLay)])
# softMax = tf.keras.layers.Softmax
model = tf.keras.Sequential([baseModel, glob2D, predLay])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

for layer in baseModel.layers[:100]:
	layer.trainable = False

history = model.fit_generator(dataSetTrain.repeat().shuffle(164), epochs=20, steps_per_epoch=steps_per_epoch)