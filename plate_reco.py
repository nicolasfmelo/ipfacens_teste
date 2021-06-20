import cv2
import numpy as np
import imutils
import glob
from imutils.object_detection import non_max_suppression

img  = cv2.imread("plate.png", 0)

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
    closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return closing

def gradient_img(img: np.array, k=(5, 5))-> np.array:
    kernel = np.ones(k, np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient

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

blackhat = blackhat_morph(img, (30, 30))
blackhat = dilate_img(blackhat)
cnts = contour(blackhat)
(x, y, w, h) = bbox_adjust(cnts[0], axis=1, torelance=10)
teste = img[y:y + h, x:x + w]
teste_black_hat = blackhat_morph(teste, (25, 25))

t_cnts = contour(teste_black_hat)
cv2.imshow("teste Blackhat", teste_black_hat)
cv2.waitKey(0)
cv2.destroyAllWindows()

(x, y, w, h) = cv2.boundingRect(t_cnts[0])
teste_b = teste[y:y + h, x:x + w]
cv2.imshow("dilated", teste_b)
cv2.waitKey(0)
cv2.destroyAllWindows()

dilated_plate = teste_b.copy()
_, thresh  = cv2.threshold(dilated_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Final1", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = contour(thresh)
(x, y, w, h) = cv2.boundingRect(cnts[0])
thresh = thresh[y:y + h, x:x + w]
#thresh = open_img(thresh, k=(3, 3))
cv2.imshow("Final2", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh = cv2.bitwise_not(thresh)
cv2.imshow("image3", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

def pre_process_template(template: np.array)->np.array:
    template = cv2.resize(template, (90, 90))
    template = cv2.bitwise_not(template)
    cnts = contour(template)
    (x, y, w, h) = bbox_adjust(cnts[0], axis=0, torelance=10)
    template = template[y:y + h, x:x + w]
    template = dilate_img(template)
    return template

t_img = ("caracteres/M.png")

template = cv2.imread(t_img, 0)
template = pre_process_template(template)
print(template.shape)
cv2.imshow("Template",template)
cv2.waitKey(0)
cv2.destroyAllWindows()

tH, tW = template.shape[:2]
result = cv2.matchTemplate(thresh, template,
cv2.TM_CCOEFF_NORMED)
(yCoords, xCoords) = np.where(result >= 0.5)
clone = thresh.copy()
rects = []
for (x, y) in zip(xCoords, yCoords):
	rects.append((x, y, x + tW, y + tH))

pick = non_max_suppression(np.array(rects))
for (startX, startY, endX, endY) in pick:
	cv2.rectangle(clone, (startX, startY), (endX, endY),
		(255, 0, 0), 3)
cv2.imshow("match_template", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
