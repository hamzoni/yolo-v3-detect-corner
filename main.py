import os
import uuid

import cv2
from services.UtilService import UtilService as util
from services.DetectService import DetectService

OUTPUT_FOLDER = './images/out'
TEST_FOLDER = '/home/qt/datasets/classifier-data/far_capture'
FOLDER_TMP = './images/failed_scan'

detector = DetectService('x3e53')

detector.init()

cap = cv2.VideoCapture(0)


def read_files():
    files = os.listdir(TEST_FOLDER)
    for f in files:
        fp = TEST_FOLDER + '/' + f
        image = cv2.imread(fp)
        image = util.resize(image, 800)

        results = detector.detect(image)

        for rect in results[0]:
            r = rect[0]
            cv2.rectangle(image, r[0], r[3], (255, 255, 0), 2)

        if len(results) == 2:
            centroids, img_warp = detector.transform(results[1], image)
            if img_warp is not None:
                cv2.imshow('img_warp', img_warp)
                for pos in centroids:
                    cv2.circle(image, tuple(pos), 3, (0, 0, 255), 5)
                    detector.draw(image, pos)

        cv2.imshow('image', image)
        k = cv2.waitKey(0)

        if k == 27:
            return
        elif k == 13:
            cv2.imwrite('outs/' + str(uuid.uuid4()) + '.jpg', image)
    cv2.destroyAllWindows()


def read_webcam():
    while True:
        _, image = cap.read()

        image = util.resize(image, 800)

        results = detector.detect(image)

        if len(results) != 0:
            try:
                for rect in results[0]:
                    r = rect[0]
                    cv2.rectangle(image, r[0], r[3], (255, 255, 0), 2)
            except:
                print('error')
        if len(results) == 2 or results[1] is not None:
            centroids, img_warp = detector.transform(results[1], image)
            if img_warp is not None:
                cv2.imshow('img_warp', img_warp)
                for pos in centroids:
                    cv2.circle(image, tuple(pos), 3, (0, 0, 255), 5)
                    detector.draw(image, pos)

        cv2.imshow('image', image)
        k = cv2.waitKey(1)

        if k == 27:
            return
        elif k == 32:
            cv2.waitKey(0)

    cv2.destroyAllWindows()


read_webcam()
# read_files()
