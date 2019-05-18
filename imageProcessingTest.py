import matplotlib.pyplot as plt
from imageProcessing import imageProcessing

class imageProcessingTest:
	def __init__(self):
		self.imageProcessing = imageProcessing()

	def _plot(self, image, lbl):
		plt.imshow(image)
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])
		plt.title(str(lbl))
		plt.pause(5)

	def _test1(self,n):
		images = self.imageProcessing.buildData("train")
		for img, lbl in images.take(n):
			print img
			print lbl
			self._plot(img, lbl)

		# image = self.imageProcessing.loadProcessImage(self.imageProcessing+"/train")

	def _test2(self):
		images = self.imageProcessing.buildData("train")
		print images

	# def _test3(self):

imp = imageProcessingTest()
imp._test2()