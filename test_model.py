import argparse

import cv2
import imutils
import numpy as np
from imutils import contours
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from utils.captchahelper import preprocess

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--input', required=True, help='Path to input directory of images.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to input model.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading pre-trained network...')
model = load_model(arguments['model'])

image_paths = list(paths.list_images(arguments['input']))
image_paths = np.random.choice(image_paths, size=(10,), replace=False)

for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]  # Sorts contours from left to right as they appear in the image.

    output = cv2.merge([gray] * 3)
    predictions = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y - 5: y + h + 5, x - 5: x + w + 5]

        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    print(f'[INFO] Captcha: {"".join(predictions)}')
    cv2.imshow('Output', output)
    cv2.waitKey()
