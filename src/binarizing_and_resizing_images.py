import os
from PIL import Image                                          
import os, sys   

root_path = "C:/Users/user/Box/Hossein Courses/CS543/project_porposal/synthetic/raw3/image_1_raw/"
output = "C:/Users/user/Box/Hossein Courses/CS543/project_porposal/synthetic/image_4/"

dirs = os.listdir(root_path)                                       

for item in dirs:
    if os.path.isfile(root_path + item):
        im = Image.open(root_path + item)
        f, e = os.path.splitext(root_path + item)
        imResize = im.resize((448,448), Image.ANTIALIAS)
        imResize.save(output + 'd' + item , quality = 30)

import os
from PIL import Image                                          
import os, sys  
import cv2

root_path = "C:/Users/user/Box/Hossein Courses/CS543/project_porposal/synthetic/raw3/image_1_masks_raw/"
output = "C:/Users/user/Box/Hossein Courses/CS543/project_porposal/synthetic/image_4_masks/"

dirs = os.listdir(root_path)                                       

for item in dirs:
    if os.path.isfile(root_path + item):
        im = cv2.imread(root_path + item, cv2.IMREAD_GRAYSCALE)
        im = cv2.threshold(im, 249.5, 250, cv2.THRESH_BINARY_INV)[1]
        f, e = os.path.splitext(root_path + item)
        imResize = cv2.resize(im, (448, 448)) 
        cv2.imwrite(output + 'd' +  item,imResize)