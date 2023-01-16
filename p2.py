import os
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

gtspath = "./materiales/gt/"
imgspath = "./materiales/img/"
hpath = "./exp/hues/"
thrspath = "./exp/thres/"
miscpath = "./exp/misc/"
"""
Códigos de conversión de colores
  RGB2HSV = 41
  BGR2HSV = 40
  RGB2HLS = 53
  BGR2HLS = 52
"""
codes = {
  "bgr2hsv":40,
  "rgb2hsv":41,
  "bgr2hls":52,
  "rgb2hls":53
}

# Rn usando bgr2hls y funciona mejor sobre todo en las 2 últimas
# Aplicar umbralizado adaptativo

# Ostu -> threshold muy básico -> umbral adaptativo

##### Funciones auxiliares de cargado/registro de imágenes #####

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

def load():
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

##### Funciones auxiliares de manipulación de imágenes #####

# def gaussImg(img,n):
#   return cv.GaussianBlur(img,(n,n),0)

def gaussImgs(imgs,n):
  gs = []
  for img in imgs:
    gs.append(cv.GaussianBlur(img,(n,n),0))
  return gs

# def getThres(imgs, rango=[]):
#   thres = []
#   for img in imgs:
#     if rango==[]:
#       lim1, lim2 = (img.min()+img.max())/3, 255
#     else:
#       lim1, lim2 = rango
#     thr = cv.threshold(img, lim1, lim2, cv.THRESH_BINARY)[1]
#     thres.append(thr)
#   return thres

def getThres1(img, lim, div):
  if lim==0:
    lim = (img.min()+img.max())/div
  return cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1]

def getThres(imgs, lim, div):
  thres = []
  for img in imgs:
    if lim==0:
      lim = (img.min()+img.max())/div
    thr = cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1]
    thres.append(thr)
  return thres

"""
Con BRG2HSV -> thresh[80/90,115/120]
"""

def otsus(imgs):
  ts=[]
  for i in imgs:
    v=filters.threshold_otsu(i)
    t.append(cv.threshold(i,v,255,cv.THRESH_BINARY)[1])
  return t

def getThresOtsu(imgs):
  thres = []
  for img in imgs:
    thr = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | 
      cv.THRESH_OTSU)[1]
    thres.append(thr)
  return thres

def getKernel(n):
  return np.ones((n,n), dtype=np.uint8)

def getOpening(img, n):
  return cv.morphologyEx(img, cv.MORPH_OPEN, getKernel(n))

def getOpenings(imgs, n):
  openings = []
  for img in imgs:
    openings.append(getOpening(img, n))
  return openings

def getErosion(img, n):
  return cv.erode(img, getKernel(n), iterations=1)

def getErosions(imgs, n):
  erosions = []
  for img in imgs:
    erosions.append(getErosion(img, n))
  return erosions

def getDilate(img, n):
  return cv.dilate(img, getKernel(n), iterations=1)

def getDilatations(imgs, n):
  dilatations = []
  for img in imgs:
    dilatations.append(getDilate(img, n))
  return dilatations

"""
Erosión con kernels bajos

"""

def getMedians(imgs, n):
  ms = []
  for i in imgs:
    ms.append(cv.medianBlur(i,n))
  return ms


def getEval(img, gt, metr):
  """
  Las imágenes deben ser binarias.
  """
  m, n = img.shape
  vp, vn, fp, fn = 0,0,0,0
  for i in range(m):
    for j in range(n):
      if (img[i,j]==1):
        if (gt[i,j]==1):
          vp+=1
        else:
          fp+=1
      else:
        if (gt[i,j]==1):
          fn+=1
        else:
          vn+=1
  if ("sens" in metr):
    return vp/(vp+fn)
  elif ("esp" in metr):
    return vn/(vn+fp)
  elif ("prec" in metr):
    return vp/(vp+fp)
  elif ("sim" in metr):
    p, s = vp/(vp+fp), vp/(vp+fn)
    return 1-(math.sqrt(((1-p)**2)+((1-s)**2))/math.sqrt(2))
  elif ("frac" in metr):
    return 1-((fp+fn)/gt.sum())
  elif ("cos" in metr):
    return cosine_similarity(img, gt)
  else:
    return [vp, fp, vn, fn]


"""
Kernels:
kv3 = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype='uint8')
kv4 = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0]], dtype='uint8')
kv5 = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]], dtype='uint8')
kh3 = np.array([[0,0,0],[1,1,1],[0,0,0]], dtype='uint8')
kh4 = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0]], dtype='uint8')
kh5 = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]], dtype='uint8')
"""


def getHues(imgs, code):
  hues = []
  for img in imgs:
    hues.append(cv.cvtColor(img, codes[code])[:,:,0])
  return hues

def compare(i1, i2):
  """
  Función que compara dos imágenes (np.array) y guarda sus scores
  según las métricas de F1 y Jaccard.
  """
  array1, array2 = i1.flatten(),i2.flatten()
  scores = {}
  scores['f1-micro'] = f1_score(array1, array2, average='micro')
  scores['f1-macro'] = f1_score(array1, array2, average='macro')
  scores['f1-none'] = f1_score(array1, array2, average=None)
  scores['jc-micro'] = jaccard_score(array1, array2, average='micro')
  scores['jc-macro'] = jaccard_score(array1, array2, average='macro')
  scores['jc-none'] = jaccard_score(array1, array2, average=None)
  return scores

def listCompare(l1, l2):
  scores = {}
  for i in range(len(l1)):
    scores[i] = compare(l1[i], l2[i])
  return scores

def invertList(l):
  result = []
  for e in l:
    result.append(255-e)
  return result

def blendLists(l1, l2, weights=[]):
  """
  Fusiona imágenes de dos listas distintas, ponderadas según weights.
  Las listas deben tener las mismas dimensiones.
  """
  result = []
  if weights==[]:
    p1, p2 = 0.5, 0.5
  else:
    p1, p2 = weights
  for i in np.arange(len(l1)):
    result.append(l1[i]*p1+l2[i]*p2)
  return result

###################

def exp1():
  imgs, gts = load()
  hues = getHues(imgs, "bgr2hsv")
  gs = gaussImgs(hues, 5)
  # ts = getThres(gs, [80,255], 0)
  ts = getThres(gs, 80, 0)
  tt = cv.threshold(gs[0], 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
  return imgs, gts, hues, gs, ts, tt

def exp2():
  imgs, gts = load()
  hues = getHues(imgs, "bgr2hsv")
  hues2 = getHues(imgs, "rgb2hsv")
  gs = gaussImgs(hues, 5)
  gs2 = gaussImgs(hues2, 5)
  # ts = getThres(gs, [80,255], 0)
  ts = getThres(gs, 80, 0)
  ts2 = invertList(getThres(gs2, 50, 0))
  tt = cv.threshold(gs[0], 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
  return imgs, gts, hues, gs, ts, ts2

###################

imgs, gts = load()
hues = getHues(imgs, "bgr2hsv")
hues2 = getHues(imgs, "rgb2hsv")
gs = gaussImgs(hues, 5)
gs2 = gaussImgs(hues2, 5)
# ts = getThres(gs, [80,255], 0)
ts = getThres(gs, 80, 0)
ts2 = invertList(getThres(gs2, 50, 0))

es = getErosions(gs, 3)
ms = getMedians(gs, 3)