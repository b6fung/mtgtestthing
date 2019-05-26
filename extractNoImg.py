import os
from shutil import copyfile
# import sys
a = ""
for i, v in enumerate(os.listdir("")):
	p = os.listdir("/"+v)[0]
	c = "/"+v+"/"+p
	copyfile(c,a+str(i)+".jpg")