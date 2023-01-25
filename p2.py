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
Cambios a posteriori:
  - Arreglado mecanismo para guardar las imágenes segmentadas
  - Arreglada función para registrar las métricas de evaluación
  - Arreglado workaround en getEval 
    -> el nivel de la imagen segmentada tiene que ser 254, no 1
  - Arreglados cálculos erróneos de sensibilidad y precisión


"""


gtspath = "./materiales/gt/"
imgspath = "./materiales/imag/"
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

metr_keys = ["sens","esp","prec","sim","frac","sobresegm","subsegm","f1","jacc"]
umbral_sobre = 0.7
umbral_sub = 0.5

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

# def save_img(img, name, path):
#   cv.imwrite(path+name+".png", img)

def save_imgs(imgs, names, path):
  for i in range(len(imgs)):
    # save_img(imgs[i],names[i],path)
    cv.imwrite(path+names[i]+".png", imgs[i])

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

def getHues(imgs, code):
  hues = []
  for img in imgs:
    if code==0:
      hues.append(img[:,:,0])
    else:
      hues.append(cv.cvtColor(img, codes[code])[:,:,0])
  return hues

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
  
  if inv:
    return cv.threshold(img, lim, 255, cv.THRESH_BINARY_INV)[1]
  else:
    return 

def getThres(imgs, lim, div, inv):
  """
  Aplica umbralización a todas las imágenes en una lista.
   - lim = umbral
   - div = división de la media (max+min)
  """
  thres = []
  for img in imgs:
    if lim==0:
      lim = (int(img.min())+int(img.max()))/div
    thres.append(cv.threshold(img, lim, 255, cv.THRESH_BINARY)[1])
  return thres

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

def getOpening(img, n):
  return cv.morphologyEx(img, cv.MORPH_OPEN, getKernel(n))

def getOpenings(imgs, n):
  openings = []
  for img in imgs:
    openings.append(getOpening(img, n))
  return openings

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

def getHitsOrMissesTotal(img):
  """
  Aplica la transformada Hit-or-Miss con kernels de distintas dimensiones
  y grosores, en verticala, horizontal y en ambas diagonales.
  El resultado de cada transformada se suma al anterior para no
  descartar partes de la imagen.
  """
  hm = np.zeros(img.shape, dtype=np.uint8)
  for i in [3,4,5,6,7,8]:
    for j in [1,2,3]:
      kernel = getHMKernel(i,j)
      diag = np.diag(np.ones(i))
      hm = hm + getHitOrMiss(img,kernel) + getHitOrMiss(img, kernel)
      # hm += hm + getHitOrMiss(img, diag) + getHitOrMiss(img, np.flip(diag)) 
  return hm

def getHMList(imgs):
  hms = []
  for img in imgs:
    hms.append(getHitsOrMissesTotal(img))
  return hms

#################### EVALUACIÓN ####################

def getEval(img, gt):
  """
  Las imágenes deben ser binarias.
  """
  m, n = img.shape
  vp, vn, fp, fn = 0,0,0,0
  for i in range(m):
    for j in range(n):
      if (img[i,j]==254):
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
  # print("FP: ",fp,"\tVP: ",vp,"\tFN: ",fn)
  ev["sens"]=vp/(vp+vn)
  ev["esp"]=vn/(vn+fp)
  ev["prec"]=vp/(vp+fp)
  if (vp+fp)!=0:
    p = vp/(vp+fp)
  else:
    p = 0
  if (vp+fn)!=0:
    s = vp/(vp+fn)
  else:
    s = 0
  ev["sim"]=1-(math.sqrt(((1-p)**2)+((1-s)**2))/math.sqrt(2))
  ev["frac"]=1-((fp+fn)/gt.sum())
  ev["cos"]=cosine_similarity(img, gt)
  ev["sobresegm"]=(ev["prec"]>umbral_sobre)
  ev["subsegm"]=(ev["esp"]>umbral_sub)
  array1, array2 = img.flatten(), gt.flatten()
  ev["f1"]=f1_score(array1, array2, average=None)
  ev["jacc"]=jaccard_score(array1, array2, average=None)
  return ev

def getEvs(imgs, gts):
  evs = []
  for i in range(len(imgs)):
    evs.append(getEval(imgs[i],gts[i]))
  return evs

def getRoadLength(img):
  """
  Devuelve la longitud total de la carretera (en píxeles)
  presente en la imagen.
  -> número de píxeles blancos en la imagen.
  """
  return img.sum()

###################

def getVals(d):
  vals = []
  for k in metr_keys:
    vals.append((k,d[k]))
  return vals

def processReports(file, names, reports):
  file = Path(file)
  file.touch(exist_ok=True)
  with open(file,"a") as f:
    for i in range(len(reports)):
      f.write("Métricas para "+names[i]+"\n")
      for key,value in getVals(reports[i]):
        f.write(key+": "+str(value)+"\n")


def processImgs(imgs,gts):
  """
  Función principal. Segementa las imágenes de referencia y
  las evalúa con ayuda de los ground truths.
  Devuelve un array con las imágenes segmentadas y un diccionario
  con las métricas para cada imagen.
  """
  # Pasamos las imágenes a escala de grises
  hues1, hues2 = getHues(imgs,"bgr2hsv"), getHues(imgs,"")
  # Suavizamos las imágenes con un filtro gaussiano
  gs1, gs2 = gaussImgs(hues1,5), gaussImgs(hues2,5)
  ts = getThres(gs1,0,2.5,0)
  # Apertura + erosión para eliminar elementos pequeños 
  oss = getOpenings(ts,4)
  lines = getErosions(oss,2)
  # Eliminamos elementos demasiado grandes como para ser carreteras
  lines2 = []
  for line in lines:
    lines2.append(line-getOpening(line,15))
  # Aplicamos Transf. Hit-or-Miss para reducir y definir el 
  # número de líneas de la figura
  segmentations = getHMList(lines2)
  reports = getEvs(segmentations,gts)
  return segmentations, reports


# def main():
parser = argparse.ArgumentParser()
parser.add_argument('source_dir', help="carpeta con las imágenes a segmentar.",
  type = dir_path)
parser.add_argument('target_dir', help="carpeta en la que guardar las imágenes segmentadas.",
  type = dir_path)
parser.add_argument('-v', help="mostrar las imágenes segmentadas por pantalla.",
  required=False, action="store_true")
parser.add_argument('-m', help="mostrar resultados de métricas de evaluación por pantalla.",
  required=False, action="store_true")
args = parser.parse_args()
# Nombre de la carpeta que contiene las subcrpetas con las imágenes
# y los ground truths
src = args.source_dir
dest = args.target_dir+"/results/"
# Cargamos las imágenes
imgs,gts = load_imgs(src+"/imag/",src+"/gt/")
# Procesamos las imágenes y las evaluamos contra sus ground truths.
segms, reports = processImgs(imgs,gts)
# Creamos un array de nombres para las imágenes y las guardamos
names = getNames(len(segms),"segmentacion_img")
if not os.path.exists(dest):
  os.mkdir(dest)
# save_imgs(names,segms,dest)
for i in range(len(segms)):
  # save_img(imgs[i],names[i],path)
  cv.imwrite(dest+names[i]+".png", segms[i])
# Volcamos los resultados en un fichero
processReports(dest+"results.txt",names, reports)

if args.v:
  for i in range(len(segms)):
    print("Longitud de la carretera:\t",getRoadLength(segms[i]),"píxeles.")
    if args.m:
      print("Métricas de evaluación de la segmentación:")
      print(reports[i])
    show(segms[i])



# if __name__ == "__main__":
#   main()
