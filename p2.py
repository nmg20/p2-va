import os
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import skimage.morphology as sm


##############################
"""
Dudas:
  - Aplicar gaussiana antes de operadores morfológicos(?)
  - Aplicar primero dilatación para preservar las carreteras y luego erosión
    -> Cierre
  - Umbralización adaptativa(?)
  - Umbralización binaria parametrizada por la media(?)
  - Aplicar thinning/skeletization


"""
##############################

# Rn usando bgr2hls y funciona mejor sobre todo en las 2 últimas
# Ostu -> threshold muy básico -> umbral adaptativo
# Erosión con kernels bajos para no deformar demasiado las carreteras

gtspath = "./materiales/gt/"
imgspath = "./materiales/img/"
hpath = "./exp/hues/"
thrspath = "./exp/thres/"
miscpath = "./exp/misc/"

codes = {
  "bgr2hsv":40,
  "rgb2hsv":41,
  "bgr2hls":52,
  "rgb2hls":53,
  "":0
}

umbral_sobre = 0.7
umbral_sub = 0.5
# Hit or miss
# gradiente morfologico antes de umbralizar
# Aplicar umbralizado adaptativo -> zonas mas grandes
# Umbralizacion con histeresis

"""
PROBAR:
  - Umbralización con histéresis
  - Umbralizado adaptativo
  - Gradiente morfológico
"""

##### Funciones auxiliares de cargado/registro de imágenes #####

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

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

def load_dir(path):
  imgs = []
  for i in os.listdir(path):
    imgs.append(cv.imread(path+i,cv.IMREAD_GRAYSCALE)/255.)
  return imgs

def load_imgs(imgspath, gtspath):
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

#################### SUAVIZADO ####################

def gaussImgs(imgs,n):
  gs = []
  for img in imgs:
    gs.append(cv.GaussianBlur(img,(n,n),0))
  return gs

def getMedians(imgs, n):
  ms = []
  for i in imgs:
    ms.append(cv.medianBlur(i,n))
  return ms

#################### THRESHOLDING ####################

def getThres1(img, lim, div, inv):
  if lim==0:
    lim = (int(img.min())+int(img.max()))/div
  if inv:
    return cv.threshold(img, lim, 255, cv.THRESH_BINARY_INV)[1]
  else:
    return cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1]

def getThres(imgs, lim, div, inv):
  """
  Aplica umbralización a todas las imágenes en una lista.
   - lim = umbral
   - div = división de la media (max+min)
  """
  thres = []
  for img in imgs:
    # if lim==0:
    #   lim = (int(img.min())+int(img.max()))//div
    # print("MIN: ",img.min(),"\tMAX: ",img.max(),"\tLIM: ",lim)
    # thr = cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1]
    thres.append(getThres1(img,lim,div,inv))
  return thres

"""
Con BRG2HSV -> thresh[80/90,115/120]
"""

def getAdaptiveThres(img, n):
  # Probar dividiendo la imagen en una matriz 5x5
  # Gaussiano
  m = img.shape[0]//n
  if m%2==0:
    m+=1
  # img2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
  #   cv.THRESH_BINARY, (m//5)+1, 2)
  img2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV, m, 6)
  return img2

def acot(img, lims):
  r = np.zeros(img.shape, dtype=np.uint8)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i,j]>lims[0] and img[i,j]<lims[1]:
        r[i,j]=255.
        # r[i,j]=img[i,j]
  return r

def getAcots(imgs):
  a2 = []
  for i in imgs:
    l = i.min()+i.max()
    lims = [l/2,l*2/3]
    a2.append(acot(i,lims))
  return a2

############## OPERACIONES MORFOLÓGICAS ##############

def getKernel(n):
  return np.ones((n,n), dtype=np.uint8)

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

def getOpening(img, n):
  return cv.morphologyEx(img, cv.MORPH_OPEN, getKernel(n))

def getOpenings(imgs, n):
  openings = []
  for img in imgs:
    openings.append(getOpening(img, n))
  return openings

def getClosing(img, n):
  return cv.morphologyEx(img, cv.MORPH_CLOSE, getKernel(n))

def getClosings(imgs, n):
  closings = []
  for img in imgs:
    closings.append(getClosing(img, n))
  return closings

# def cm(matrix):
#   return np.flip(matrix,axis=1)

# def getDiagKernel(size,n):
#   o = np.ones((1,size))
#   a0,a1,a2 = np.diag(o), np.diag(o,k=1), np.diag(o,k=-1)
#   a=a1
#   while n>1:



def getHMKernel(size, n):
  # Arrys horizontales
  z = np.zeros((size,1), dtype="int")
  o = np.ones((size,1), dtype="int")
  while n>1:
    o = np.concatenate((o,o), axis=1)
    n-=1
  return np.concatenate((z, np.concatenate((o,z),axis=1)), axis=1)

def getHitOrMiss(img,kernel):
  return cv.morphologyEx(img, cv.MORPH_HITMISS,kernel)

def getHitsOrMisses(img,sizes=[],ns=[]):
  """
  Aplica la transformada Hit-or-Miss recusivamente con kernels horizontales
  de tamaño y densidad variantes.
    -img: np.array de tipo uint8
  """
  # sizes = 3,4,5 | ns = 1,2,3
  # hs = {}
  hs = []
  for s in sizes:
    for n in ns:
      # hs[s]={n:getHitOrMiss(img, getHMKernel(s,n))}
      hs.append(getHitOrMiss(img, getHMKernel(s,n))+getHitOrMiss(img, getHMKernel(s,n).T))
  return hs

def getHMList(imgs, kernel):
  hms = []
  for img in imgs:
    hms.append(getHitOrMiss(img,kernel))
  return hms


############## SKELETIZATION(?) ##############

def sk2Array(sk):
  arr = np.zeros(sk.shape)
  for i in range(sk.shape[0]):
    for j in range(sk.shape[1]):
      if sk[i,j]==True:
        arr[i,j]=255.
  return arr


#################### EVALUACIÓN ####################

def getEval(img, gt):
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
  ev = {}
  ev["sens"]=vp/(vp+fn)
  ev["esp"]=vn/(vn+fp)
  ev["prec"]=vn/(vn+fp)
  p, s = vp/(vp+fp), vp/(vp+fn)
  ev["sim"]=1-(math.sqrt(((1-p)**2)+((1-s)**2))/math.sqrt(2))
  ev["frac"]=1-((fp+fn)/gt.sum())
  ev["cos"]=cosine_similarity(img, gt)
  ev["sobresegm"]=(ev["prec"]>umbral_sobre)
  ev["subsegm"]=(ev["esp"]>umbral_sub)
  array1, array2 = img.flatten(), gt.flatten()
  ev["f1"]=f1_score(array1, array2, average=None)
  ev["jacc"]=jaccard_score(array1, array2, average=None)
  return ev

def getRoadLength(img):
  """
  Devuelve la longitud total de la carretera (en píxeles)
  presente en la imagen.
  -> número de píxeles blancos en la imagen.
  """
  return g.sum()

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
    if code==0:
      hues.append(img[:,:,0])
    else:
      hues.append(cv.cvtColor(img, codes[code])[:,:,0])
  return hues

# def compare(i1, i2):
#   """
#   Función que compara dos imágenes (np.array) y guarda sus scores
#   según las métricas de F1 y Jaccard.
#   """
#   array1, array2 = i1.flatten(),i2.flatten()
#   scores = {}
#   scores['f1-micro'] = f1_score(array1, array2, average='micro')
#   scores['f1-macro'] = f1_score(array1, array2, average='macro')
#   scores['f1-none'] = f1_score(array1, array2, average=None)
#   scores['jc-micro'] = jaccard_score(array1, array2, average='micro')
#   scores['jc-macro'] = jaccard_score(array1, array2, average='macro')
#   scores['jc-none'] = jaccard_score(array1, array2, average=None)
#   return scores

# def listCompare(l1, l2):
#   scores = {}
#   for i in range(len(l1)):
#     scores[i] = compare(l1[i], l2[i])
#   return scores

# def invertList(l):
#   result = []
#   for e in l:
#     result.append(255-e)
#   return result

# def blendLists(l1, l2, weights=[]):
#   """
#   Fusiona imágenes de dos listas distintas, ponderadas según weights.
#   Las listas deben tener las mismas dimensiones.
#   """
#   result = []
#   if weights==[]:
#     p1, p2 = 0.5, 0.5
#   else:
#     p1, p2 = weights
#   for i in np.arange(len(l1)):
#     result.append(l1[i]*p1+l2[i]*p2)
#   return result

###################

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



###################

imgs, gts = load_imgs(imgspath,gtspath)
hues = getHues(imgs, "bgr2hsv")
hues2 = getHues(imgs, "rgb2hsv")
gs = gaussImgs(hues, 5)
gs2 = gaussImgs(hues2, 5)
# ts = getThres(gs, [80,255], 0)
ts = getThres(gs, 80, 0,0)
# ts2 = invertList(getThres(gs2, 50, 0))

# Erosiones
es = getErosions(gs, 3)
# Aperturas
ops = getOpenings(gs, 5)
# Cierres
cs = getClosings(gs, 5)

# Suavizado de medias
ms = getMedians(gs, 3)



hs1, hs2 = getHues(imgs, "bgr2hsv"), getHues(imgs, "")
gs1, gs2 = gaussImgs(hs1, 5), gaussImgs(hs2, 5)
ms1, ms2 = getMedians(hs1, 5), getMedians(hs2, 5)
gst, mst = [], []
for i in range(len(hs1)):
  gst.append(gs1[i]+gs2[i])
  mst.append(ms1[i]+ms2[i])

csg, csm = getClosings(gst,5), getClosings(gsm,5)
osg, osm = getOpenings(gst,5), getOpenings(gsm,S5)

tsg0, tsm0 = getThres(gst,190,0,0), getThres(mst,190,0,0)



ts50 = load_dir("./exp/exp3/thresh50/")

def processImgs(imgs,gts):
  # Pasamos las imágenes a escala de grises
  hues = getHues(imgs,"bgr2hsv")
  # Suavizamos las imágenes con un filtro gaussiano
  gs = gaussImgs(hues,5)

  # Eliminar ruido con apertura

  return result, report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source_dir', help="carpeta con las imágenes a segmentar.",
    type = dir_path)
  parser.add_argument('target_dir', help="carpeta en la que guardar las imágenes segmentadas.",
    type = dir_path)
  parser.add_argument('-v', help="mostrar las imágenes segmentadas por pantalla.",
    action='visualize')
  parser.add_argument('-o', help="mostrar resultados de métricas de evaluación por pantalla.",
    action='report')
  args = parser.parse_args()
  # Nombre de la carpeta que contiene las subcrpetas con las imágenes
  # y los ground truths
  src = args.source_dir
  dest = args.target_dir
  imgs,gts = load_imgs(src+"/img",src+"/gt")
  segms, report = processImgs(imgs,gts)

  print("")
  for i in segms:
    print(getRoadLength(i))


