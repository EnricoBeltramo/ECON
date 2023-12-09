
from diffusers.utils import load_image
import uuid
import os
import shutil


from segmentation.segmenter import Segmenter

segManager = Segmenter()

img_url = "overture-creations-5sI6fQgYIuo5.png"
#img_url = "0a64d9c7ac4a86aa0c29195bc6f55246.jpg"
#img_url = "baf3c945fa6b4349c59953a97740e70f.jpg"
img_url = "examples/baf3c945fa6b4349c59953a97740e70f.jpg"
img_url = "examples/test2.jpg"
#img_url = "examples/test.png"
img_url = "examples/test.png"
img_url = "examples/test6.png"

basename = str(uuid.uuid4())

SEED = 6
SAVE_DIR = 'results/resultsSeg2'
DILATE_STEP = 2

GUIDANCE_SCALE=7.5
INFERENCE_STEPS=20  # steps between 15 and 30 work well for us
STRENGTH=1-1e-2  # make sure to use `strength` below 1.0

LOOPS = 1


shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR,exist_ok=True)

def getPathImage(extension):
  fileName = basename +  '_' + extension + '.jpg'
  return os.path.join(SAVE_DIR, fileName)

def saveImage(image,extension):
  fileName = basename +  '_' + extension + '.jpg'
  image.save(getPathImage(extension)) 
  return fileName

def saveSAMImage(segManager,extension):
  segManager.saveMaskSAM(getPathImage(extension)) 

imageToParse = load_image(img_url)


imgCropped, imgSkin, imgPerson, imgSkinSAM = segManager.cropAndMask(imageToParse)

#saveSAMImage(segManager,'segmentationSAM')
saveImage(imgCropped,'cropped')
saveImage(imgSkin,'skin')
saveImage(imgPerson,'person')
saveImage(imgSkinSAM,'skinSAM')

print('done')