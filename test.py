import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score

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

##### Funciones auxiliares de cargado/registro de imágenes #####

###################
"""
Con BRG2HSV -> thresh[80/90,115/120]
"""
def exp1():
  # imgs, gts = load_imgs()
  hues = getHues(imgs, "bgr2hsv")
  gs = gaussImgs(hues, 5)
  # ts = getThres(gs, [80,255], 0)
  ts = getThres(gs, 80, 0)
  tt = cv.threshold(gs[0], 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
  return imgs, gts, hues, gs, ts, tt

def exp2():
  # imgs, gts = load()
  hues = getHues(imgs, "bgr2hsv")
  hues2 = getHues(imgs, "rgb2hsv")
  gs = gaussImgs(hues, 5)
  gs2 = gaussImgs(hues2, 5)
  # ts = getThres(gs, [80,255], 0)
  ts = getThres(gs, 80, 0)
  ts2 = invertList(getThres(gs2, 50, 0))
  tt = cv.threshold(gs[0], 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
  return imgs, gts, hues, gs, ts, ts2

def exp3():
  i=imgs[0][:,:,0]
  g = gaussImgs([i],5)[0]
  # t = getThres1(g,100,0)
  t = getThres1(g,0,2.5,0)
  o = getOpening(t,5)
  c = getClosing(t,5)
  # shown([i,g,t,o])
  go = getOpening(g,5)
  gc = getClosing(g,5)
  tgo = getThres1(go, 100,0,0)
  tgc = getThres1(gc, 100,0,0)
  oss = getOpening(go,15)
  shown([tgo,tgc,go-oss])

def exp_rgb():
  hues = getHues(imgs,"rgb2hsv")
  gs = gaussImgs(hues, 5)
  oss = getOpenings(gs,5)
  hm = getHitOrMiss(oss[0],getHMKernel(5,2))

def exp_acot():
  hues = getHues(imgs,"rgb2hsv")
  gs = gaussImgs(hues, 5)
  a2 = getAcots(gs)
  oss = getOpenings(a2,3)-getOpenings(a2,11)
  cs = getClosings(oss,5)
  shown(cs)

imgs, gts = load_imgs(imgspath,gtspath)
hues = getHues(imgs, "bgr2hsv")
hues2 = getHues(imgs, "rgb2hsv")
# gs = gaussImgs(hues, 5)
# gs2 = gaussImgs(hues2, 5)
# # ts = getThres(gs, [80,255], 0)
# ts = getThres(gs, 80, 0,0)
# # ts2 = invertList(getThres(gs2, 50, 0))
# # Erosiones
# es = getErosions(gs, 3)
# # Aperturas
# ops = getOpenings(gs, 5)
# # Cierres
# cs = getClosings(gs, 5)
# # Suavizado de medias
# ms = getMedians(gs, 3)

hs1, hs2 = getHues(imgs, "bgr2hsv"), getHues(imgs, "")
gs1, gs2 = gaussImgs(hs1, 5), gaussImgs(hs2, 5)
ms1, ms2 = getMedians(hs1, 5), getMedians(hs2, 5)
gst, mst = [], []
for i in range(len(hs1)):
  gst.append(gs1[i]+gs2[i])
  mst.append(ms1[i]+ms2[i])
csg, csm = getClosings(gst,5), getClosings(mst,5)
osg, osm = getOpenings(gst,5), getOpenings(mst,5)
tsg0, tsm0 = getThres(gst,190,0,0), getThres(mst,190,0,0)

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

def getThres(imgs, lims, div):
  thres = []
  for img in imgs:
    if lims==[]:
      lim1 = (img.min()+img.max())/div
      lim2 = 255
    else:
      lim1, lim2 = lims
    thr = cv.threshold(img, lim1, lim2, cv.THRESH_BINARY)[1]
    thres.append(thr)
  return thres

"""
Con BRG2HSV -> thresh[80/90,115/120]
"""

def getOpenings(imgs, kernel, n):
  if kernel==[]:
    kernel = np.ones((n,n), dtype='uint8')
  openings = []
  for img in imgs:
    o = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    openings.append(o)
  return openings

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

def listcompare(l1, l2):
  scores = {}
  for i in range(len(l1)):
    scores[i] = compare(l1[i], l2[i])
  return scores

def main():
  imgs, gts = load()
  
def supresionNoMax(mag):
  # Find the neighbouring pixels (b,c) in the rounded gradient direction
# and then apply non-max suppression
  M, N = mag.shape
  Non_max = np.zeros((M,N), dtype= np.uint8)
  for i in range(1,M-1):
      for j in range(1,N-1):
         # Horizontal 0
          if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
              b = mag[i, j+1]
              c = mag[i, j-1]
          # Diagonal 45
          elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
              b = mag[i+1, j+1]
              c = mag[i-1, j-1]
          # Vertical 90
          elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
              b = mag[i+1, j]
              c = mag[i-1, j]
          # Diagonal 135
          elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
              b = mag[i+1, j-1]
              c = mag[i-1, j+1]          
          # Non-max Suppression
          if (mag[i,j] >= b) and (mag[i,j] >= c):
              Non_max[i,j] = mag[i,j]
          else:
              Non_max[i,j] = 0
  return Non_max

def umbralizacionHisteresis(Non_max, lowThreshold, highThreshold):
  M, N = Non_max.shape
  out = np.zeros((M,N), dtype= np.uint8)

  strong_i, strong_j = np.where(Non_max >= highThreshold)
  zeros_i, zeros_j = np.where(Non_max < lowThreshold)
  weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

  out[strong_i, strong_j] = 255
  out[zeros_i, zeros_j ] = 0
  out[weak_i, weak_j] = 75
  M, N = out.shape
  for i in range(1, M-1):
      for j in range(1, N-1):
          if (out[i,j] == 75):
              if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
                  out[i, j] = 255
              else:
                  out[i, j] = 0
  return out



if __name__ == "__main__":
  main()