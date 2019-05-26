import tensorflow as tf
from tensorflow import contrib
from tqdm import tqdm
tfe = contrib.eager
# import sciki
from imageProcessing import imageProcessing
IMAGE_SHAPE = (224, 224,3)
baseModel = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
										  include_top=False, 
										  weights='imagenet')
for layer in baseModel.layers[:100]:
	layer.trainable = False
# baseModel.trainable = False

glob2D = tf.keras.layers.GlobalAveragePooling2D()
predLay = tf.keras.layers.Dense(1)

num_epoch = 20
BATCH_SIZE = 10
dataObj = imageProcessing()
dataSetTrain = dataObj.buildData("train")
dataSetTrain = dataSetTrain.repeat().shuffle(164)
# dataSetTrain = dataSetTrain.shuffle(buffer_size = dataObj.numExp)
dataSetTrain = dataSetTrain.batch(BATCH_SIZE)
# dataSetTest = dataObj.buildData("test")

# dataSetTest = dataSetTest.shuffle(buffer_size = dataObj.numExp)
# dataSetTest = dataSetTest.batch(BATCH_SIZE)
# print baseModel.summary()
model = tf.keras.Sequential([baseModel, glob2D, predLay])
# print imgBat
def loss(model, input, labels):
	pred = model(input)
	labels = tf.expand_dims(labels, axis=1)
	return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits = pred)

def grad(moded, input, target):
	with tf.GradientTape() as tape:
		lossVal = loss(model, input, target)
	return lossVal, tape.gradient(lossVal, model.trainable_variables)


optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
step = tf.Variable(0)

for epoch in range(num_epoch):
	epoch_acc = tfe.metrics.Accuracy()
	print epoch
	for x, y in dataSetTrain:
		loss_val, grads = grad(model, x, y)
		optimizer.apply_gradients(zip(grads, model.trainable_variables), step)
		if epoch > 4:
			print model(x)
			print y
		epoch_acc(tf.argmax(model(x), axis=1, output_type=tf.int64), y)
		# print epoch_acc.result()



# printNode = tf.Print(predLay, [tf.shape(predLay)])
# softMax = tf.keras.layers.Softmax

# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])




# history = model.fit_generator(dataSetTrain.repeat().shuffle(164), epochs=20, steps_per_epoch=steps_per_epoch)