import numpy as np
import cv2
from time import time


# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    #print(cutWindow)
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# Simple tracker
class BaseTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self._interp_factor = 0.1  # model updating rate
        self._tmpl_sz = np.array([50, 50])  #the fixed model size
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [left_up_x,left_up_y,width,height]
        self._tmpl = None  # our model
        self._scale_pool = [0.985,0.99 ,0.995,1.0,1.005,1.01,1.015]

    def getFeatures(self, image, roi, needed_size):
        roi = list(map(int, roi))  # ensure that everything is int
        z = subwindow(image, roi, cv2.BORDER_REPLICATE)  # sample a image patch

        if z.shape[1] != needed_size[0] or z.shape[0] != needed_size[1]:
            z = cv2.resize(z, tuple(needed_size))  # resize to template size

        if z.ndim == 3 and z.shape[2] == 3:
            FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        elif z.ndim == 2:
            FeaturesMap = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5

        return FeaturesMap

    def measure(self, a, b):
        """
        This is the measure function you need to code
        you should use self.measure in self.track to simplify your code
        """
        return (np.linalg.norm(a-b))**2

    def track(self, search_region, img):
        """
        this is the main place you need to code
        please check ln.117 to see what's the input of this function
        """
        ind_x, ind_y, ind_s = 0, 0, 1.0

        min_sim=1000000

        for s in self._scale_pool:  #选择合适的scale
            #print('s',search_region)

            valid_region = [search_region[0],search_region[1],search_region[2]*s,search_region[3]*s]
            limit(valid_region, [0, 0, img.shape[1], img.shape[0]])  # 取不超出图片范围的有效区域,注意这里shape[0]是h,shape[1]是w
            #print(valid_region)

            # 计算剪裁掉的边界部分按search_region与tmpl的比例缩放后的大小
            margin_x = (valid_region[0] - search_region[0]) / (search_region[2]*s) * 3/2 * self._tmpl_sz[0]
            margin_y = (valid_region[1] - search_region[1]) / (search_region[3]*s) * 2 * self._tmpl_sz[1]

            #计算有效区域按search_region与tmpl的比例缩放以后的大小
            resize = np.array([int(valid_region[2] / (search_region[2]*s) * 3/2 * self._tmpl_sz[0]),
                               int(valid_region[3] / (search_region[3]*s) * 2 * self._tmpl_sz[1])])

            # 获取整个搜索区域的特征
            feature = self.getFeatures(img, valid_region, resize)

            for i in range(feature.shape[0] - self._tmpl_sz[1]):  # feature是以高*宽的格式存储的，因此shape[0]是h,shape[1]是w
                for j in range(0,feature.shape[1] - self._tmpl_sz[0],3):
                    #滑动窗口提取self._tmpl_sz大小的特征
                    target_img = feature[i:i + self._tmpl_sz[1], j:j + self._tmpl_sz[0]]
                    similar = self.measure(target_img, self._tmpl) #计算相似度
                    if (similar < min_sim):
                        min_sim = similar
                        ind_x = j + margin_x   #ind_x是相对于search_region[0]的偏移值，因此需要加上被剪裁掉的边界部分
                        ind_y = i + margin_y
                        ind_s = s


        return ind_x, ind_y, ind_s

    def update_model(self, x, train_interp_factor):
        """
        This is the model update function you need to code
        """
        self._tmpl = train_interp_factor*x + (1-train_interp_factor)*self._tmpl


    def init(self, roi, image):
        self._roi = list(map(int, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, self._roi, self._tmpl_sz)
        #print(self._tmpl)

    def update(self, image):
        # some check boundary here
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2
        # center position of our target
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.
        # we double the searching region compared to the selected region
        search_rect = [cx - self._roi[2]/2, cy - self._roi[3], self._roi[2] * 3/2, self._roi[3] * 2]

        # the delta in search region
        loc_pos = self.track(search_rect, image)
        print(loc_pos)

        # delta_x and delta_y we want to estimate
        delta = (np.array(loc_pos[:2]) -np.array([0,50]) / 2)
        #因为search_roi是扩展了两倍大小的，并且计算feature时做了resize，因此计算相对_roi的位置要减去_tmpl_sz/2

        # scale between the search_roi and our template
        scale = loc_pos[2] * np.array(search_rect[2:]).astype(float) / (np.array([self._roi[2] * 3/2, self._roi[3] * 2]))
        # back to the original size
        delta = delta * scale

        # add the delta to original position
        self._roi[0] = self._roi[0] + delta[0]
        self._roi[1] = self._roi[1] + delta[1]
        self._roi[2] = self._roi[2] * loc_pos[2]
        self._roi[3] = self._roi[3] * loc_pos[2]

        # some check boundary here
        if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        # update the template
        x = self.getFeatures(image, self._roi, self._tmpl_sz)  # new observation
        self.update_model(x, self._interp_factor)

        return self._roi
