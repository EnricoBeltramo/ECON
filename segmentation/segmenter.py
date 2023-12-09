from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
import cv2
import os

try:
    import clip  # for linear_assignment

except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements(
        "git+https://github.com/openai/CLIP.git"
    )  # required before installing lap from source
    import clip


from .fastsam import FastSAM, FastSAMPrompt


SEED = 6
DILATE_STEP = 2

# 0 Background
# 1 Cappello 0,73 0,68
# 2 Capelli 0,91 0,82
# 3 Occhiali da sole 0,73 0,63
# 4 Soprabiti 0,87 0,78
# 5 Gonna 0,76 0,65
# 6 Pantaloni 0,90 0,84
# 7 Abito 0,74 0,55
# 8 Cintura 0,35 0,30
# 9 Scarpa sinistra 0,74 0,58
# 10 Scarpa destra 0,75 0,60
# 11 Faccia 0,92 0,85
# 12 Gamba sinistra 0,90 0,82
# 13 Gamba destra 0,90 0,81
# 14 Braccio sinistro 0,86 0,74
# 15 Braccio destro 0,82 0,73
# 16 Borsa 0,91 0,84
# 17 Sciarpa 0,63 0,29


class Segmenter:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )

        # load model
        self.sammodel = FastSAM(
            model="./weights/FastSAM-x.pt",
            device="cuda",
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )

        self.annotations = None
        self.prompt_process = None
        self.clip_model, self.preprocess = None, None

    def resizeImage(self, image):
        # Ottieni le dimensioni originali dell'immagine
        width, height = image.size

        # Determina la dimensione del lato del quadrato
        side_length = min(width, height)

        # Calcola le coordinate del punto in alto a sinistra per il taglio
        left = (width - side_length) // 2
        top = 0

        # Calcola le coordinate del punto in basso a destra per il taglio
        right = left + side_length
        bottom = top + side_length

        # Esegui il taglio dell'immagine per ottenere il quadrato
        square_image = image.crop((left, top, right, bottom))

        image_square_resize = square_image.resize((1024, 1024))

        return image_square_resize

    def samSegmentation(self, image):
        input = image.convert("RGB")
        everything_results = self.sammodel(input)
        # masks: prompt_process.results[0].masks.data (numbers of mask, h, w)
        prompt_process = FastSAMPrompt(input, everything_results, device="cuda")
        annotations = prompt_process.everything_prompt()

        prompt_process.plot(
            annotations=annotations,
            output_path="testSAM/mask.jpg",
            bboxes=None,
            points=None,
            point_label=None,
            withContours=True,
            better_quality=True,
        )

        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())

        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            annotations[i] = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
            )

        # Sort annotations based on area.
        areas = np.sum(annotations, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        annotationsSorted = annotations[sorted_indices]

        mixedMask = (annotationsSorted != 0).argmax(axis=0)

        maxId = mixedMask.max()

        h = annotations.shape[1]
        w = annotations.shape[2]

        splittedMask = []

        for ind in range(0, maxId + 1):
            mask = np.zeros((h, w))
            mask[mixedMask == ind] = 1
            splittedMask.append(mask)

        self.annotations = splittedMask
        self.prompt_process = prompt_process

        return splittedMask

    def saveMaskSAM(self, pathImage):
        annotations = self.annotations
        prompt_process = self.prompt_process
        prompt_process.plot(
            annotations=annotations,
            output_path=pathImage,
            bboxes=None,
            points=None,
            point_label=None,
            withContours=True,
            better_quality=True,
        )

    def baseSegmentation(self, image):
        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_seg = pred_seg.cpu().numpy().astype(np.uint8)

        return pred_seg

    def baseSkin(self, segmentation):
        pred_seg = segmentation.copy()

        # 0 Background
        # 1 Cappello 0,73 0,68
        # 2 Capelli 0,91 0,82
        # 3 Occhiali da sole 0,73 0,63
        # 4 Soprabiti 0,87 0,78
        # 5 Gonna 0,76 0,65
        # 6 Pantaloni 0,90 0,84
        # 7 Abito 0,74 0,55
        # 8 Cintura 0,35 0,30
        # 9 Scarpa sinistra 0,74 0,58
        # 10 Scarpa destra 0,75 0,60
        # 11 Faccia 0,92 0,85
        # 12 Gamba sinistra 0,90 0,82
        # 13 Gamba destra 0,90 0,81
        # 14 Braccio sinistro 0,86 0,74
        # 15 Braccio destro 0,82 0,73
        # 16 Borsa 0,91 0,84
        # 17 Sciarpa 0,63 0,29

        # no mask all parts that arent body parts
        nomask = [11, 12, 13, 14, 15]
        # mask all clothes, hair and backgroun
        mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]

        pred_seg[np.isin(segmentation, nomask)] = 255
        pred_seg[np.isin(segmentation, mask)] = 0

        return pred_seg

    def splitSingleAreas(self, mask):
        # Trova i contorni nella maschera
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lista per memorizzare le maschere separate
        splitted = []

        # Crea una maschera per ogni contorno trovato
        for contorno in contours:
            # Crea una nuova maschera vuota
            maschera_temp = np.zeros_like(mask)

            # Disegna il contorno sulla maschera
            cv2.drawContours(maschera_temp, [contorno], -1, 1, thickness=cv2.FILLED)

            # Aggiungi la maschera alla lista
            splitted.append(maschera_temp)

        return splitted

    def orMasks(self, masks):
        # Inizializza una maschera risultante con tutti i pixel impostati a 0
        res = np.zeros_like(masks[0])

        # Applica l'OR logico su tutte le maschere
        for mask in masks:
            res = np.logical_or(res, mask)

        return np.uint8(res) * 255

    def boundingBox(self,mask):
        # Trova i contorni nella maschera
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Trova il contorno con l'area maggiore
        max_contour = max(contours, key=cv2.contourArea)

        # Calcola il bounding box per il contorno più grande
        x, y, w, h = cv2.boundingRect(max_contour)

        return [y, y + h, x, x + w]

    def mergeSkinMask(self, image, maskSAM, maskSkinBase):
        th = 0.5
        thprob = 0.85

        skinArrayFound = []

        splittedMaskBase = self.splitSingleAreas(maskSkinBase)
        splittedFeaturesBase = [None] * len(splittedMaskBase)

        for j, mask in enumerate(maskSAM):
            maskedImage = self.applyMask(image, mask)
            maskedImage.save(
                os.path.join(
                    "results/resultsSeg2", format(0, f".{3}f") + str(j) + ".jpg"
                )
            )

        for i, maskBase in enumerate(splittedMaskBase):
            maskedImage = self.applyMask(image, maskBase)
            maskedImage.save(
                os.path.join(
                    "results/resultsSeg2", format(0, f".{3}f") + str(i) + "_base.jpg"
                )
            )

            for j, mask in enumerate(maskSAM):
                # if i == 2 and j == 8:
                #     print("here")
                maskInt = np.int8(mask)

                # Calcola l'intersezione (AND logico) delle due maschere
                intersection = np.logical_and(maskInt, maskBase)

                # Conta i punti che sono 1 in entrambe le maschere
                count_intersection = np.sum(intersection)

                # Conta i punti che sono 1 nella maschera sam
                countMaskSAM = np.sum(mask)

                # Calcola la percentuale
                percentInSAM = (
                    0 if countMaskSAM == 0 else count_intersection / countMaskSAM
                )

                # se la mask sam è quasi completamente inclusa
                if percentInSAM > th:

                    yi,ye,xi,xe = self.boundingBox(maskBase)
                    imageCrop = image.crop((xi, yi, xe, ye))

                    maskCrop = mask[yi:ye,xi:xe]
                    maskedImageSAM = self.applyMask(imageCrop, maskCrop)
                    featuresMaskSAM = self.extractClipFeatures(maskedImageSAM)

                    maskCropBase = maskBase[yi:ye,xi:xe]
                    maskedImageBase = self.applyMask(imageCrop, maskCropBase)
                    featuresMaskBase = self.extractClipFeatures(maskedImageBase)
                    splittedFeaturesBase[i] = featuresMaskBase

                    probs = float((featuresMaskBase @ featuresMaskSAM.T))

                    if probs > thprob:
                        maskedImageSAM.save(
                            os.path.join(
                                "results/resultsSeg2",
                                format(probs, f".{3}f") + str(j) + "_found.jpg",
                            )
                        )
                        skinArrayFound.append(mask)

        finalMask = self.orMasks(skinArrayFound)

        return finalMask

    def extractClipFeatures(self, image):
        with torch.no_grad():
            if self.clip_model is None:
                self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")

            preprocessed_images = self.preprocess(image).to("cuda")
            stacked_images = torch.stack([preprocessed_images])
            image_features_ref = self.clip_model.encode_image(stacked_images)
            image_features_ref /= image_features_ref.norm(dim=-1, keepdim=True)
        return image_features_ref

    def basePerson(self, segmentation):
        pred_seg = segmentation.copy()
        # mask background
        mask = [0]

        pred_seg[:] = 255
        pred_seg[np.isin(segmentation, mask)] = 0

        return pred_seg

    def dilateErodemask(self, segmentation, N=5):
        mask = segmentation.copy()

        # Crea un kernel per la dilatazione (puoi personalizzarne la dimensione)
        kernel = np.ones((N, N), np.uint8)

        # Esegui la dilatazione sulla maschera
        mask = cv2.dilate(mask, kernel, iterations=DILATE_STEP)

        mask = cv2.erode(mask, kernel, iterations=DILATE_STEP)

        return mask

    def applyMask(self, image, mask, color=None):
        imageCV = self.pilToCV(image)

        imageMasked = imageCV.copy()

        if color is None:
            color = [128, 128, 128]
        imageMasked[mask == 0] = color

        return self.cvToPil(imageMasked)

    def pilToCV(self, imagePIL):
        imageCv = np.array(imagePIL)

        return imageCv

    def cvToPil(self, imagecv):
        imagePIL = Image.fromarray(imagecv)

        return imagePIL

    def cropAndMask(self, image):
        imgCropped = self.resizeImage(image)
        maskBase = self.baseSegmentation(imgCropped)
        maskSAM = self.samSegmentation(imgCropped)

        maskSkinBase = self.baseSkin(maskBase)
        maskPersonBase = self.basePerson(maskBase)
        maskSkinSAM = self.mergeSkinMask(imgCropped, maskSAM, maskSkinBase)

        imgSkinBase = self.applyMask(imgCropped, maskSkinBase)
        imgPersonBase = self.applyMask(imgCropped, maskPersonBase)
        imgSkinSAM = self.applyMask(imgCropped, maskSkinSAM)

        return imgCropped, imgSkinBase, imgPersonBase, imgSkinSAM
