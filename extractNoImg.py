import os
from shutil import copyfile
# import sys
a = "/Users/BrandonFung/MTGDS/train/no/"
for i, v in enumerate(os.listdir("/Users/BrandonFung/Downloads/101_ObjectCategories")):
	p = os.listdir("/Users/BrandonFung/Downloads/101_ObjectCategories/"+v)[0]
	c = "/Users/BrandonFung/Downloads/101_ObjectCategories/"+v+"/"+p
	copyfile(c,a+str(i)+".jpg")