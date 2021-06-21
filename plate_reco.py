import cv2
import numpy as np
import imutils
import glob
from skimage.metrics import structural_similarity as ssim
import os

img  = cv2.imread("placa.png", 0)
original = cv2.imread("placa.png", 1)

def remove_background(img: np.array, kernel=35)-> np.array:
    median = cv2.medianBlur(img, kernel)
    cleaned = cv2.subtract(median, img)
    cleaned = cv2.bitwise_not(cleaned)
    return cleaned

def blackhat_morph(img: np.array, kernel: tuple)-> np.array:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
    return blackhat

def contour(img: np.array)-> list:
    _, thresh  = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    return cnts

def dilate_img(img: np.array, k=(5, 5))-> np.array:
    kernel = np.ones(k,np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def erode_img(img: np.array, k=(5,5))-> np.array:
    kernel = np.ones(k,np.uint8)
    erode = cv2.erode(img,kernel,iterations = 1)
    return erode

def open_img(img: np.array, k=(5, 5))-> np.array:
    kernel = np.ones(k,np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    return closing

def close_img(img: np.array, k=(5, 5))-> np.array:
    kernel = np.ones(k,np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing

def gradient_img(img: np.array, k=(5, 5))-> np.array:
    kernel = np.ones(k, np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient

def padding(img: np.array, padd=5, color=[255, 255, 255]):
    constant= cv2.copyMakeBorder(img,padd,padd,padd,padd,cv2.BORDER_CONSTANT,value=color)
    return constant

def bbox_adjust(bbox: np.array, axis=0, torelance=0)-> tuple:
    (x, y, w, h) = cv2.boundingRect(bbox)
    t_w = w*(torelance/100)
    t_h = h*(torelance/100)
    if axis == 0:
        w = int(t_w+w)
        return (x, y, w, h)
    elif axis ==1:
        h = int(t_h+h)
        return (x, y, w, h)
    else:
        return (x, y, w, h)


def pre_process_template(template: np.array)-> np.array:
    template = cv2.bitwise_not(template)
    cnts = contour(template)
    (x, y, w, h) = bbox_adjust(cnts[0], axis=0, torelance=10)
    template = template[y:y + h, x:x + w]
    template = dilate_img(template)
    template = padding(template, padd=5, color=[0, 0, 0])
    template = cv2.resize(template, (38, 85))
    return template

blackhat = blackhat_morph(img, (30, 30))
blackhat = dilate_img(blackhat)
cnts = contour(blackhat)
(x, y, w, h) = bbox_adjust(cnts[0], axis=1, torelance=30)
bbox = (x, y, w, h)
teste = img[y:y + h, x:x + w]

teste_black_hat = blackhat_morph(teste, (30, 30))
t_cnts = contour(teste_black_hat)
(x, y, w, h) = bbox_adjust(t_cnts[0], axis=1, torelance=30)
bbox = (bbox[0]+x, bbox[1]+y, w, h)
teste_b = teste[y:y + h, x:x + w]


dilated_plate = teste_b.copy()
_, thresh  = cv2.threshold(dilated_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cnts = contour(thresh)
(x, y, w, h) = bbox_adjust(cnts[0], axis=1, torelance=-3)
bbox = (bbox[0]+x, bbox[1]+y, w, h)
thresh = thresh[y:y + h, x:x + w]
thresh = padding(thresh, padd=5)
thresh = open_img(thresh, k=(3, 3))
tresh = close_img(thresh, k=(3,3))
cv2.imshow("Op 1", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh = cv2.bitwise_not(thresh)
cv2.imshow("Op 2", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

segmented = contour(thresh)
char_database = glob.glob("./caracteres/*png")
char_bboxes = []

for row in segmented[:7]:
    (x, y, w, h) = cv2.boundingRect(row)
    letter = thresh[y:y+h, x: x+w]
    letter_bbox = (x, y, w, h)
    letter = padding(letter, padd=5, color=[0, 0, 0])
    #letter = close_img(letter, k=(5, 5))
    letter = cv2.resize(letter, (38, 85))
    candidates = []
    for files in char_database:
        template = cv2.imread(files, 0)
        template = pre_process_template(template)
        value = ssim(letter, template)
        candidates.append((value, files))
    best_candidate = max(candidates)
    template = cv2.imread(best_candidate[1], 0)
    name = os.path.basename(best_candidate[1]).split(".")[0]
    char_bboxes.append((letter_bbox, name))
    template = pre_process_template(template)


(x, y, w, h) = bbox
cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Plate",original)
cv2.waitKey(0)
cv2.destroyAllWindows()

for box in char_bboxes:
    (x, y, w, h) = box[0]
    coord = (bbox[0]+x, bbox[1]+y, w, h)
    name = box[1]
    (x, y, w, h) = coord
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(original, name, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

cv2.imshow("Final",original)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./final.png", original)