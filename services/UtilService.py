import base64
import math
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class UtilService:
    @staticmethod
    def resize(image, size):
        height, width, channel = image.shape

        ratio = width / height

        if width > height:
            if width > size:
                width = size
                height = width / ratio
        else:
            if height > size:
                height = size
                width = ratio * height

        image = cv2.resize(image, (int(width), int(height)))
        return image

    @staticmethod
    def four_points_transform(image, rect):
        print(rect)
        (tl, tr, br, bl) = rect

        wa = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        wb = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        mw = max(int(wa), int(wb))

        ha = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        hb = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        mh = max(int(ha), int(hb))

        dst = np.array([
            [0, 0],
            [mw - 1, 0],
            [mw - 1, mh - 1],
            [0, mh - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (mw, mh))
        return warped

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def intersects(self, other):
        return not (
                self.top_right.x < other.bottom_left.x or
                self.bottom_left.x > other.top_right.x or
                self.top_right.y < other.bottom_left.y or
                self.bottom_left.y > other.top_right.y)

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        if img is None:
            return
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def order_points(pts):
        if len(pts) < 4:
            return

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def timeit(method):

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()

            print('%r %2.2f sec' % (method.__name__, te - ts));
            return result

        return timed

    @staticmethod
    def plot_images(np_images, titles=[], columns=2, figure_size=(24, 18)):
        count = len(np_images)
        rows = math.ceil(count / columns)

        fig = plt.figure(figsize=figure_size)
        subplots = []
        for index in range(count):
            subplots.append(fig.add_subplot(rows, columns, index + 1))
            if len(titles):
                subplots[-1].set_title(str(titles[index]))

            image = cv2.cvtColor(np_images[index], cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.axis('off')

        plt.show()

    @staticmethod
    def mat2pil(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    @staticmethod
    def arr2mat(image):
        image = np.array(image)
        return image[:, :, ::-1].copy()

    @staticmethod
    def to_image(image):
        image_cv = np.array(image)
        return image_cv[:, :, ::-1].copy()

    @staticmethod
    def readb64(base64_string):
        nparr = np.fromstring(base64.b64decode(base64_string), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
