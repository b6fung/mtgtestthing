import pathlib
import tensorflow as tf
tf.enable_eager_execution()

class imageProcessing:
	def __init__(self):
		self.path = "/Users/BrandonFung/MTGDS"
		self.HEIGHT = 224
		self.WIDTH = 224
		self.numExp = 0

	def _loadImages(self, path):
		image = tf.io.read_file(path)
		return image

	def _processImage (self, image):
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.resize(image,[self.HEIGHT, self.WIDTH])
		# image = (image / 127.5) - 1
		image = image / 255.0
		return image

	def loadProcessImage(self, path):
		return self._processImage(self._loadImages(path))

	def buildData(self,option):
		all_image_paths = []
		all_labels = []
		str2Num = {"yes":1, "no":0}
		for i in ["yes","no"]:
			dataRoot = pathlib.Path(self.path+"/"+option+"/"+i)
			imagePaths = list(dataRoot.glob('**/*'))
			self.numTrainExp = len(imagePaths)
			all_image_paths = all_image_paths + [str(x) for x in imagePaths]
			all_labels = all_labels + [str2Num[i]]*self.numTrainExp
		path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
		# print all_image_paths
		images_ds = path_ds.map(lambda x: self.loadProcessImage(x))
		labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int64))
		full_ds = tf.data.Dataset.zip((images_ds, labels_ds))
		full_ds.make_one_shot_iterator()
		return full_ds


