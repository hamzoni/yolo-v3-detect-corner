from services.UtilService import UtilService as util
from utils.datasets import *
from utils.models import *
from utils.utils import *
from pandas import DataFrame
from sklearn.cluster import KMeans

IMAGE_SIZE = 512
IMAGE_SIZE = 416
WEIGHT_CONFIG = '/home/qt/models/yolo'
WEIGHT_MODEL = '/home/qt/models/yolo'
IMAGE_SAMPLE = './images'
CONF_THRES = 0.3
NMS_THRES = 0.5
CLASSES = ['corner']


class DetectService:

    def __init__(self, version):
        self.model = None
        self.device = None
        self.version = version

    def init(self):
        self.device = torch_utils.select_device()
        torch.backends.cudnn.benchmark = False

        self.model = Darknet(WEIGHT_CONFIG + '/' + self.version + '.cfg', IMAGE_SIZE)
        load_darknet_weights(self.model, WEIGHT_MODEL + '/' + self.version + '.weights')

        self.model.fuse()
        self.model.to(self.device).eval()

    @staticmethod
    def load(img0, height):
        # img0 = cv2.flip(img0, 1)
        img, *_ = letterbox(img0, new_shape=height)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img, img0

    @staticmethod
    def draw(image, pos):
        if len(pos) < 4:
            return
        # pos = list(map(lambda x: np.asarray(x), pos))
        pos = np.int0(pos)
        pos = util.order_points(pos)
        cv2.rectangle(image, tuple(pos[1]), tuple(pos[3]), (0, 255, 0), 3)

    @util.timeit
    def detect(self, image):
        # Get classes and colors
        img, im0 = DetectService.load(image, IMAGE_SIZE)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, CONF_THRES, NMS_THRES)[0]

        results = {}

        if det is not None and len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for *x, conf, cls_conf, cls in det:
                idx = int(cls.item())

                if idx not in results:
                    results[idx] = []

                x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
                c1, c2, c3, c4 = (x1, y1), (x2, y2), (x1, y2), (x2, y1)

                image_result = image[c1[1]:c2[1], c1[0]:c2[0]]
                coordinates = sorted([c1, c2, c3, c4], key=lambda k: (k[0], k[1]))
                results[idx].append([coordinates, image_result, x, im0])
                # util.plot_one_box(x, im0, (0, 0, 255))

        # cv2.imshow('image', im0)

        return results

    def getCornerCentroid(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = int(sum(_x_list) / _len)
        _y = int(sum(_y_list) / _len)
        return _x, _y

    def collectVertices(self, rects):
        vertices = list(map(lambda x: list(self.getCornerCentroid(list(map(lambda y: list(y), x[0])))), rects))
        if len(vertices) == 0:
            return vertices

        return vertices

        centroids = vertices

        dataX = list(map(lambda p: p[0], vertices))
        dataY = list(map(lambda p: p[1], vertices))

        Data = {
            'x': dataX,
            'y': dataY
        }

        df = DataFrame(Data, columns=['x', 'y'])

        try:
            kmeans = KMeans(n_clusters=4).fit(df)
            centroids = kmeans.cluster_centers_
            centroids = list(map(lambda p: (int(p[0]), int(p[1])), centroids))
            return centroids
        except:
            return centroids

    def transform(self, rects, image):

        for rect in rects:
            r = rect[0]
            cv2.rectangle(image, r[0], r[3], (0, 255, 0), 2)

        # for rect in rects:
        centroids = self.collectVertices(rects)
        if len(centroids) > 0:
            approx = util.order_points(np.int0(centroids))
            img_warp = util.four_points_transform(image, approx)
            return centroids, img_warp
