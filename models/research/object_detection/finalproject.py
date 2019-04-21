from PIL import Image
import numpy as np
import cv2 as cv
import os, sys, time
import tensorflow as tf
import math

sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

############################# ^^ initializes tensorflow ^^######################

width = 1280
height = 720

#this switches the coordinates if needed
def rectify(h):
  if(len(h)== 4):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
   
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew
  else:
    return h

#mouse even is detected and if it's a left click the detect_card function is called
def select_card(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("mouse click x: %5d y: %5d" %(x, y))
        detect_card(x,y,param)

#tensor card detection happens here
def detect_card(x,y,cap):
  # Perform the actual detection by running the model with the image as input
  frame_expanded = np.expand_dims(cap, axis=0)
  (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: frame_expanded})

  #fun little line of code that made finding out how many objects were detected so much easier
  cards_detected = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
  cardsLength = len(cards_detected)
  print(cardsLength)

  #have to check if the click was on a card or not
  #if on a card then try to match it with database
  if(cardsLength > 0):
    smallBox = np.squeeze(np.asarray(boxes))
    for i, box in zip(range(cardsLength),smallBox):
      ymin = int(box[0]*height)
      xmin = int(box[1]*width)
      ymax = int(box[2]*height)
      xmax = int(box[3]*width)
      print("xmin:{}, ymin:{}, xmax:{}, ymax:{}".format(xmin,ymin,xmax,ymax))
      print("top left1:{},{}".format(xmin,ymin))
      print("bottom right1:{},{}".format(xmax,ymax))
      if(xmin<=x<=xmax and ymin<=y<=ymax):
        print("match card")
        match_card(xmin,ymin,xmax,ymax,cap)
      else:
        print("mouse click not in bounds")




  
def match_card(xmin,ymin,xmax,ymax,cap):
    start_time = time.time()
    dst = cap[ymin:ymax, xmin:xmax]
    #dst = cap
    #cv.imshow('dst',dst)
    kernel = np.ones((5,5),np.uint8)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9,9), 0)
    erode = cv.erode(blur, kernel,iterations =2)
    retval, thresh = cv.threshold(erode, 100, 255,cv.THRESH_BINARY_INV)
    img, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[0]

    #contour approximation
    peri = cv.arcLength(cnt, True)
    approx = rectify(cv.approxPolyDP(cnt, 0.015*peri, True))
    #pts1 = np.float32([approx[0],approx[3],approx[1],approx[2]])
    pts2 = np.float32([[0,0],[300,0],[300,420],[0,420]])
    M = cv.getPerspectiveTransform(approx, pts2)
    warp = cv.warpPerspective(dst, M, (300,420))
    art = warp[48:234, 26:274]

    path = os.path.join('E:', os.sep, 'FinalProjectImages','resized')
    dirs = os.listdir( path )

    Images = []
    kp = []
    des = []
    final = []
    matches = []
    highest = []
    keys = False

    #check if Images have been filled, if not then loop
    #if filled then skip to next section
    if not Images:
        print("images loop")
        for image in dirs:
            Images.append(cv.imread(os.path.join(path,image),1))

        sift = cv.xfeatures2d.SIFT_create(50)
        bf =cv.BFMatcher()
        #srcArt = cv.imread(art,1)
        #dstArt = srcCP[48:234, 26:274]
        (kpArt, desArt) = sift.detectAndCompute(art,None)


        for im in Images:
            if not keys:
                dst = im[48:234, 26:274]
                (key, desc) =sift.detectAndCompute(dst,None)
                kp.append(key)
                des.append(desc)
            match = (bf.knnMatch(desArt, desc, k=2))
            matches.append(match)
            x = 0
            good = []
            for m,n in match:
                if m.distance < 0.75*n.distance:
                    x = x +1;
                    good.append([m])
            final.append(good)
            highest.append(x)

    best_match = max(highest)
    print (best_match)
    i = highest.index(best_match)
    print ("this took", time.time() - start_time, "to run")
    cv.imshow('img',Images[i])
    keys = True
    final = None
    matches = None
    highest = None
    i=None
    #result = cv.drawMatchesKnn(Images[i],kp[i],srcCP,kpCP, final[i], None, flags=2)


def main():
    cap = cv.VideoCapture(0)
    ret = cap.set(3,width)
    ret = cap.set(4,height)
    
    while(cap.isOpened()):
        ret, src = cap.read()
        cv.setMouseCallback('src',select_card, src)
        
        cv.imshow('src',src)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("quit")
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        logging.getLogger(__name__).exception("Program terminated")
        raise
