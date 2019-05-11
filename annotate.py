import argparse
import os

import cv2
import imutils
from imutils import paths

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--input', required=True, help='Path to input directory of images.')
argument_parser.add_argument('-a', '--annotation', required=True, help='Path to output directory of annotatons.')
arguments = vars(argument_parser.parse_args())

image_paths = list(paths.list_images(arguments['input']))
counts = dict()

for i, image_path in enumerate(image_paths):
    print(f'[INFO] Processing image {i + 1}/{len(image_paths)}')

    try:
        # Load image, convert it to gray and pad it.
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # Threshold the image to reveal the digits.
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find contours in the image, keeping only the four largest ones.
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y - 5: y + h + 5, x - 5: x + w + 5]

            cv2.imshow('ROI', imutils.resize(roi, width=28))
            key = cv2.waitKey(0)  # The key pressed will be the label of the digit.

            if key == ord("q"):
                print("[INFO] Ignoring character.")
                continue

            key = chr(key).upper()
            dir_path = os.path.sep.join([arguments['annotations'], key])

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            count = counts.get(key, 1)
            p = os.path.sep.join([dir_path, f'{str(count).zfill(6)}.png'])
            cv2.imwrite(p, roi)

            counts[key] = count + 1

    except KeyboardInterrupt:
        print('[INFO] Manually leaving script...')
        break
    except:
        print('[INFO] Skipping image...')