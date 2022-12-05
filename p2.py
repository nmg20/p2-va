import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


gtspath = "./materiales/gt/"
imgspath = "./materiales/img/"
hpath = "./materiales/hues/"
thrspath = "./materiales/thres/"

# DUDAS:
# - Las carreteras tienen el mismo color siempre? (sombras y zonas similares)
# - El grosor de la carretera es siempre el mismo?
#     -> número de carriles
# - Resultado -> un píxel de ancho(?) -> supresión no máxima

# IDEA:
# - Tomar el color base de la carretera para extraer sólo zonas que correspondan
# - Ajustar con apertura/cierre las formas que puede tomar la carretera
# - RGB2HSV
# - BGR2HLS

def save_img(img, name, path):
  cv.imwrite(os.path.join(path,name+".png"), img)

def save_imgs(imgs, names, path):
  i = 0
  while i<len(imgs):
    save_img(imgs[i],names[i],path)
    i+=1

def umbr(image, thres):
  u = image.copy()
  u[u<thres]=0
  u[u>=thres]=1
  return u


def load_imgs():
  gts, imgs = [],[]
  # Se cargan las imágenes de ejemplo
  for img in os.listdir(imgspath):
    imgs.append(cv.imread(imgspath+img))
  # Se cargan los ground truths
  for gt in os.listdir(gtspath):
    gts.append(cv.imread(gtspath+gt))
  return imgs, gts

def show(img,rsz):
  if rsz :
    img = cv.resize(img,(500,500))
  cv.imshow("", img)
  cv.waitKey(0)
  cv.destroyAllWindows()

def shown(imgs, rsz):
  i=0
  while i<len(imgs):
    if rsz :
      img = cv.resize(imgs[i],(500,500))
    else :
      img = imgs[i]
    cv.imshow("win"+str(i),img)
    cv.waitKey(0)
    cv.destroyWindow("win"+str(i))
    i+=1

def toHsv(imgs):
  hsv = []
  for img in imgs:
    hsv.append(cv.cvtColor(img,cv.COLOR_RGB2HSV))
  return hsv

def names(n,name):
  names=[]
  for i in np.arange(0,n):
    names.append(name+str(i))
  return names

def getHues(imgs):
  hues = []
  for img in imgs:
    hues.append(cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,0])
  return hues

def ex(hue):
  ret, thr = cv.threshold(hue, 127, 255, cv.THRESH_TOZERO_INV) #Importante que sea invertido
  ret, thr = cv.threshold(thr, 50, 150, cv.THRESH_BINARY_INV)
  shown([u1,u2],1)

def getThres(hues):
  thres = []
  for hue in hues:
    ret, thr = cv.threshold(hue, 127, 255, cv.THRESH_TOZERO_INV) #Importante que sea invertido
    ret, thr = cv.threshold(thr, 50, 150, cv.THRESH_BINARY_INV)
    thres.append(thr)
  return thres


def main():
  imgs, gts = load_imgs()
  # hues = []
  # for img in imgs:
  #   hues.append(cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,0])
  hues = getHues(imgs)
  # shown(hues,0)
  # save_imgs(hues,names(len(hues),"hsv""),hpath)
  thres = getThres(hues)
  save_imgs(thres,names(len(thres),"thres"),thrspath)



if __name__ == "__main__":
  main()