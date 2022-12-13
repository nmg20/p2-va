import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


gtspath = "./materiales/gt/"
imgspath = "./materiales/img/"
hpath = "./exp/hues/"
thrspath = "./exp/thres/"
miscpath = "./exp/misc/"

# DUDAS:
# - Las carreteras tienen el mismo color siempre? (sombras y zonas similares)
# - El grosor de la carretera es siempre el mismo?
#     -> número de carriles
# - Resultado -> un píxel de ancho(?) -> supresión no máxima
# - ¿Cómo estimar longitudes?

# IDEA:
# - Tomar el color base de la carretera para extraer sólo zonas que correspondan
# - Ajustar con apertura/cierre las formas que puede tomar la carretera
# - RGB2HSV
# - BGR2HLS
"""
Approach:
  - Convertir la imagen de RGB a HSV
  - Procesar sólo el canal H
  - Umbralizarlo
  ***- Hacer opening/closing para deshacerse de patches pequeños
  - Aplicar filtro de medianas para eliminar ruido una vez umbralizado
  - cv.findContours (?)

Progreso:
  - conseguidos buenos resultados con umbralización
  - errores en la última imágen
    -> idea: hacer el threshold variable

"""

def getNames(n,name):
  names=[]
  for i in np.arange(0,n):
    names.append(name+str(i))
  return names

def save_img(img, name, path):
  if path=="":
    path=miscpath
  cv.imwrite(os.path.join(path,name+".png"), img)

def save_imgs(imgs, names, path):
  i = 0
  # names = getNames(len(imgs),name)
  if not os.path.exists(path):
    os.makedirs(path)
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
    # Cargamos las grounf truths en escala de grises
    gts.append(cv.imread(gtspath+gt, cv.IMREAD_GRAYSCALE)/255.)
  return imgs, gts

def show(img):
  img = cv.resize(img,(500,500))
  cv.imshow("", img)
  cv.waitKey(0)
  cv.destroyAllWindows()

def shown(imgs):
  i=0
  while i<len(imgs):
    img = cv.resize(imgs[i],(500,500))
    cv.imshow("win"+str(i),img)
    cv.waitKey(0)
    cv.destroyWindow("win"+str(i))
    i+=1

def toHsv(imgs):
  hsv = []
  for img in imgs:
    hsv.append(cv.cvtColor(img,cv.COLOR_RGB2HSV))
  return hsv



def getHues(imgs):
  hues = []
  for img in imgs:
    hues.append(cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,0])
  return hues

def getHues2(imgs):
  hues = []
  for img in imgs:
    hues.append(cv.cvtColor(img,cv.COLOR_RGB2HLS)[:,:,0])
  return hues

def ex(hue):
  ret, thr = cv.threshold(hue, 127, 255, cv.THRESH_TOZERO_INV) #Importante que sea invertido
  ret, thr = cv.threshold(thr, 50, 150, cv.THRESH_BINARY_INV)
  shown([u1,u2])

def gaussImg(img,n):
  return cv.GaussianBlur(img,(n,n),0)

def gaussImgs(imgs,n):
  gs = []
  for i in range(0,len(imgs)):
    gs.append(gaussImg(imgs[i],n))
  return gs

def getThres(imgs):
  thres = []
  for img in imgs:
    lim = (img.min()+img.max())/3
    thr = cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1]
    thres.append(thr)
  return thres

def getOpenings(imgs, kernel):
  if kernel==[]:
    kernel = np.ones((4,4), dtype='uint8')
  openings = []
  for img in imgs:
    o = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    openings.append(o)
  return openings

def kern(n):
  """
  Dado un número n devuelve un kernel de tamaño nxn.
  (simplifica la generación de kernels)
  """
  return np.ones((n,n), dtype='uint8')

def test(n):
  imgs, gts = load_imgs()
  huesv = getHues(imgs)
  huesl = getHues2(imgs)
  thv = getThres(huesv)
  thl = getThres(huesl)
  # shown([imgs[n],huesv[n],huesl[n],thv[n],thl[n],gts[n]])
  return imgs, gts, huesv, huesl, thl

# def exp1(imgs, n):
#   imghue = cv.cvtColor(imgs[n],cv.COLOR_RGB2HSV)[:,:,0]
#   smoothimg = cv.GaussianBlur(imghue, (5,5), 0)
#   thresh = cv.threshold(smoothimg, cv.THRESH_BINARY)[1]
#   shown([imghue,smoothimg,thresh])

def exp1(imgs,n):
  hues = getHues(imgs)
  smooths = gaussImgs(hues,5)
  threshs = getThres(smooths)
  if n>0:
    show([hues[n],smooths[n],threshs[n]])
  else:
    shown(hues+smooths+threshs)
  return hues, smooths, threshs


def main():
  imgs, gts = load_imgs()
  # hues = []
  # for img in imgs:
  #   hues.append(cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,0])
  hues = getHues(imgs)
  # shown(hues,1)
  # save_imgs(hues,names(len(hues),"hsv""),hpath)
  thres = getThres(hues)
  # save_imgs(thres,names(len(thres),"thres"),thrspath)
  # show(thres)
  openings = getOpenings(thres, [])


if __name__ == "__main__":
  main()