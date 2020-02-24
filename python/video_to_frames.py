import cv2
import os
 
# Opens the Video file
cap= cv2.VideoCapture('/home/nikhil/Documents/task/MAH04240.mp4')
path = '/home/nikhil/Documents/task/images-orig'

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # if i%10 == 0:
    source_image_output_path = os.path.join(path,'lanenet'+str(i)+'.jpg')
    cv2.imwrite(source_image_output_path, frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()