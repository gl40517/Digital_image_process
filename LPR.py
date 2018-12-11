import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import keras

DEFAUTH_CUT_THRESH = 150
DEFAUTH_ANGLE = 0
DEFAULT_GAUSSIANBLUR_SIZE = 5
SOBEL_SCALE = 1
SOBEL_DELTA = 0
SOBEL_DDEPTH = cv2.CV_16S
SOBEL_X_WEIGHT = 1
SOBEL_Y_WEIGHT = 0
DEFAULT_MORPH_SIZE_WIDTH = 25
DEFAULT_MORPH_SIZE_HEIGHT = 3
DEFAULT_ERROR = 0.8
DEFAULT_ASPECT = 3.75
DEFAULT_LETTER_THRESHOLD = 0.07
PROVINCES = ("京" ,"闽" ,"粤" ,"苏" ,"沪" ,"浙")
LETTERS_DIGITS = (
"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
"I", "O")
NUMBERS_DIGITS = (
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
"Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")


def rotateImage(img_roi):
    #----------- use Hough to auto-rotate the image#--------------------------------------

    #img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    #img_roi_canny = cv2.Canny(img_roi_gray, threshold1=200, threshold2=400)
    #lines = cv2.HoughLinesP(img_roi_canny,1,np.pi/180,20,minLineLength=20,maxLineGap=5)
    #lines1 = lines[:,0,:]#提取为二维
    #plt.figure(1)

    #line_left, line_right, line_down, line_up = None, None, None, None
    #for x1,y1,x2,y2 in lines1[:]:
        #if x1>(img_roi.shape[1]*5/6) and x2>(img_roi.shape[1]*5/6):
            #if abs(y1-y2) > img_roi.shape[0]/3:
                #line_left, line_right, line_down, line_up = x1, x2, y1, y2
    #if line_down > line_up:
        #tan = (line_down - line_up)/(line_right-line_left)
        #angle = 90 - (math.atan(tan) * 180 / math.pi)

    # ----------- use input angle to auto-rotate the image#--------------------------------------

    center = (img_roi.shape[1]//2, img_roi.shape[0]//2)
    rot = cv2.getRotationMatrix2D(center=center, angle=DEFAUTH_ANGLE, scale=1.0)
    img_roi_rotated = cv2.warpAffine(img_roi, rot, (img_roi.shape[1], img_roi.shape[0]))
    plt.imshow(img_roi_rotated[:,10:img_roi_rotated.shape[1]-10])
    plt.show()
    return img_roi_rotated

def verifySizes(mr, img_width):
    error = DEFAULT_ERROR
    aspect = DEFAULT_ASPECT
    rmin = aspect - aspect * error
    rmax = aspect + aspect * error
    angle = mr[2]
    width, height = mr[1]
    if (width == 0) or (height == 0):
        return False
    if width > height:
        if width < img_width / 4:
            return False
    elif width < height:
        if height < img_width / 4:
            return False
    r = width / height
    if r < 1:
        r = height / width

    #if (area < Min or area > Max) or (r < rmin or r > rmax):
    if r < rmin or r > rmax:
        return False
    elif -6 > angle > -84:
        return False
    else:
        return True

def cut(percentage, thresh):
    left = []
    right = []
    for i in range(len(percentage)):
        if i == 0:
            continue
        if percentage[i-1] < thresh < percentage[i]:
            left.append(i)
        if percentage[i] < thresh < percentage[i-1]:
            right.append(i)
    return left,right

img = cv2.imread('LPR.jpg')
img_blur = cv2.GaussianBlur(src=img,
                            ksize=(DEFAULT_GAUSSIANBLUR_SIZE,DEFAULT_GAUSSIANBLUR_SIZE),
                            sigmaX=0)
img_gray = cv2.cvtColor(src=img_blur,code=cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(src=img_gray,ddepth=SOBEL_DDEPTH,dx=1,
                   dy=0,ksize=3,scale=SOBEL_SCALE,delta=SOBEL_DELTA)
abs_grad_x = cv2.convertScaleAbs(src=grad_x)
grad_y = cv2.Sobel(src=img_gray,ddepth=SOBEL_DDEPTH,dx=0,
                   dy=1,ksize=3,scale=SOBEL_SCALE,delta=SOBEL_DELTA)
abs_grad_y = cv2.convertScaleAbs(src=grad_y)
grad = cv2.addWeighted(src1=abs_grad_x,alpha=SOBEL_X_WEIGHT,
                       src2=abs_grad_y,beta=SOBEL_Y_WEIGHT,gamma=0)
(retval, img_threshold) = cv2.threshold(src=grad,thresh=0,maxval=255,type=cv2.THRESH_OTSU+cv2.THRESH_BINARY)
element = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                    ksize=(DEFAULT_MORPH_SIZE_WIDTH,DEFAULT_MORPH_SIZE_HEIGHT))
img_threshold = cv2.morphologyEx(src=img_threshold,op=cv2.MORPH_CLOSE,
                                 kernel=element)
image, contours, hierarchy = cv2.findContours(image=img_threshold,
                                              mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
contours = np.array(contours)
rect = []
img_height = img.shape[0]
img_length = img.shape[1]
for i in range(len(contours)):
    mr = cv2.minAreaRect(contours[i])
    if verifySizes(mr, img_length):
        rect.append(mr)
rect = np.array(rect)
dst = np.zeros_like(img_threshold)
result = []
for i in range(rect.shape[0]):
    flag = True
    box = cv2.boxPoints(tuple(rect[i]))
    box = np.int32(box)
    for j in range(4):
        if img_height*4/5 < box[j][1] or box[j][1] < img_height*2/5:
            flag = False
            break
    Sum = 0
    for j in range(4):
        Sum += box[j][0]
    if img_length*3/5 < Sum/4 or Sum/4 < img_height*2/5:
        flag = False
    if flag:
        result.append(box)
        for j in range(4):
            cv2.line(dst, tuple(box[j]), tuple(box[(j+1) % 4]), (255,0,0), 5)
w_left, h_up = np.min(result[0],axis=0)
w_right, h_down = np.max(result[0],axis=0)
img_roi = img[h_up+5:h_down-5 , w_left:w_right]
img_roi_RGB = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
img_roi_HSV = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
blueLower = np.array([100, 80, 80])
blueUpper = np.array([124, 255, 255])
mask = cv2.inRange(img_roi_HSV, blueLower, blueUpper)
mask_percentage_vertical = (np.sum(mask,axis=0)/255) / mask.shape[0]
LPR_left = LPR_right = 0
for i in range(len(mask_percentage_vertical)):
    if mask_percentage_vertical[i] > 0.1:
        LPR_left = i
        break
for i in range(len(mask_percentage_vertical)):
    if mask_percentage_vertical[len(mask_percentage_vertical)-i-1] > 0.1:
        LPR_right = len(mask_percentage_vertical)-i
        break

img_roi = img_roi[:,LPR_left-2 : LPR_right+2]
img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
MAX = img_roi.max()
MIN = img_roi.min()
img_linear_gray = np.uint8(255 / (MAX - MIN) * img_roi - 255 * MIN / (MAX - MIN))
DEFAUTH_CUT_THRESH = eval(input('DEFAUTH_CUT_THRESH='))
ret, img_roi_threadhold = cv2.threshold(img_linear_gray, DEFAUTH_CUT_THRESH, 255, cv2.THRESH_BINARY)

plt.figure(1)
plt.imshow(img_roi_threadhold, 'gray')
plt.show()

DEFAUTH_ANGLE = eval(input('DEFAUTH_ANGLE='))
img_roi_threadhold = rotateImage(img_roi_threadhold)

percentage_vertical = (np.sum(img_roi_threadhold,axis=0)/255) / img_roi_threadhold.shape[0]
roi_left,roi_right = cut(percentage_vertical, DEFAULT_LETTER_THRESHOLD)

final_result = []
number = min(len(roi_left),len(roi_right))

for i in range(number):
    isOne = True
    if roi_right[i] - roi_left[i] > (roi_right[len(roi_right)-1] - roi_left[0]) / 11:
        final_result.append(img_roi_threadhold[:, int(roi_left[i]):int(roi_right[i])])
    else:
        aver = np.sum(percentage_vertical[roi_left[i]:roi_right[i]+1],axis=0) / (roi_right[i] - roi_left[i])
        if aver > 0.6 or aver < 0.5:
            isOne = False
        if isOne:
            final_result.append(img_roi_threadhold[:, int(roi_left[i]):int(roi_right[i])])

plt.figure(2)
for i in range(len(final_result)):
    plt.subplot(3,3,i+1)
    plt.imshow(final_result[i],'gray')
plt.show()

model_Chinese = keras.models.load_model('.\chinese_character_model.h5')
model_letter = keras.models.load_model('.\letter_model.h5')
model_number = keras.models.load_model('./number_model.h5')
predict_result = []
for i in range(len(final_result)):
    part_card = final_result[i][11:68, :]
    w = abs(part_card.shape[1] - 32) // 2
    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    part_card = cv2.resize(part_card, (32,40),interpolation=cv2.INTER_AREA)
    ret, part_card = cv2.threshold(part_card, 160, 255, cv2.THRESH_BINARY)
    if i != 0:
        kernel = np.ones((3, 3), np.uint8) #!!!!!试了一万年（一个小时）才试出来
        part_card = cv2.dilate(part_card, kernel)
    plt.imshow(part_card,'gray')
    plt.show()
    input_images = np.zeros(1280)
    width = part_card.shape[1]
    height = part_card.shape[0]
    for h in range(height):
        for w in range(width):
            if part_card[h][w] > 190:
                input_images[w + h * width] = 0
            else:
                input_images[w + h * width] = 1
    input_images = np.reshape(input_images,(1,1280))
    if i == 0:
        proba_Chinese = model_Chinese.predict_classes(input_images, verbose=0)
        predict_result.append(PROVINCES[proba_Chinese[0]])
    elif i == 1:
        proba_letter = model_letter.predict_classes(input_images, verbose=0)
        predict_result.append(LETTERS_DIGITS[proba_letter[0]])
    else:
        proba_number = model_number.predict_classes(input_images, verbose=0)
        predict_result.append(NUMBERS_DIGITS[proba_number[0]])

print(predict_result)
