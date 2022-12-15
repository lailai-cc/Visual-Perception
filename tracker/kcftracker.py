import numpy as np
import cv2
from time import time



def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    """
        Modify the index of the left-top point and width&height size in the rect so that the returned
         rectangle are within the given limit range.

        Args:
            rect: An 4 element tuple [ix, iy, w, h] to specify region of window.
                    The 4 element tuple are the coordinate of the left-top point of the rectangle,
                    and the width and height of the rectangle.
            limit: An 4 element tuple [ix, iy, w, h] to specify region of limitation.
                   The 4 element tuple are the coordinate of the left-top point of the limitation,
                   and the width and height of the limitation.
        Returns:
            res: limited rectangle [ix, iy, w, h]
    """
    if (rect[0] + rect[2] > limit[0] + limit[2]):  # right: ix + w
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):  # bottom: iy + h
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):   # left: ix
        rect[2] -= (limit[0] - rect[0])  # w = limit_x - rect_x
        rect[0] = limit[0]
    if (rect[1] < limit[1]):  # right: iy
        rect[3] -= (limit[1] - rect[1])  # h = limit_y - rect_y
        rect[1] = limit[1]
    if (rect[2] < 0):  # width
        rect[2] = 0
    if (rect[3] < 0):  # height
        rect[3] = 0
    return rect


def getBorder(original, limited):
    """
        Get the number of border pixels between the original size and the limited size

        Args:
            original: An 4 element tuple [ix, iy, w, h] to specify region of original rectangle.
                    The 4 element tuple are the coordinate of the left-top point of the rectangle,
                    and the width and height of the rectangle.
            limited: An 4 element tuple [ix, iy, w, h] to specify region of limited rectangle.
                    The 4 element tuple are the coordinate of the left-top point of the rectangle,
                    and the width and height of the rectangle.
        Returns:
            res: border pixels [l, t, r, b]
    """
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    """
        sample a patch based on the window in the image.

        Args:
            img: A 3-dimension array [w, h, 3]  of an RGB image
                 or a 2-dimension array [w, h] of a gray image
            window: An 4 element tuple [ix, iy, w, h] to specify region of window.
                    The 4 element tuple are the coordinate of the left-top point of the rectangle,
                    and the width and height of the rectangle.
            borderType: The type to fill the overflow border
        Returns:
            res: limited and filled img
    """
    cutWindow = [x for x in window]
    # Limit the size of cutWindow inside the size of img
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)  # valid w & h
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res

def cos_win(w,h):
    win_col = np.hanning(w)  # 通过hanning窗生成余弦窗
    win_row = np.hanning(h)
    mask_col, mask_row = np.meshgrid(win_col, win_row)  # 生成网格点坐标矩阵
    return mask_col*mask_row


# KCF tracker
class KCFTracker:
    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        """
        Args:
            hog: whether to use hog feature  FIXME: not implemented yet, additional task
            fixed_window (bool): whether to change the window size  FIXME: not implemented yet, additional task
            multiscale (bool): use multi-scale feature/window to track  FIXME: not implemented yet, additional task
        """

        self._interp_factor = 0.075  # model updating rate
        self._tmpl_sz = np.array([50, 50])  # the fixed model size
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [left_up_x,left_up_y,width,height]
        self._tmpl = None  # template model
        self._scale_pool = [0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015] # a scale pool that used in self.track

        self.G = None    #响应输出
        self.e = 0.0001  # regularization
        self.win= cos_win(self._tmpl_sz[0],self._tmpl_sz[1]) #余弦窗

    def getFeatures(self, image, roi, needed_size):
        """
        Extract features of a roi (region of interest) from the image and resize to a fixed size
        You should use it wherever you need a feature.
        Args:
            image: A 3-dimension array [w, h, 3]  of an RGB image
                   or a 2-dimension array [w, h] of a gray image
            roi: An 4 element tuple [ix, iy, w, h] to specify region to extract feature.
                 The 4 element tuple are the coordinate of the left-top point of the rectangle,
                 and the width and height of the rectangle.
            needed_size: The size of the return feature
        Returns:
            FeaturesMap: a needed size feature of the roi in the image
        """
        roi = list(map(int, roi))  # ensure that everything is int
        z = subwindow(image, roi, cv2.BORDER_REPLICATE)  # sample a image patch

        if z.shape[1] != needed_size[0] or z.shape[0] != needed_size[1]:
            z = cv2.resize(z, tuple(needed_size))  # resize to template size

        if z.ndim == 3 and z.shape[2] == 3:
            FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        elif z.ndim == 2:
            FeaturesMap = z

        # 将np.int数据类型转换成np.float32,方便后续计算
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5

        # 将矩形截图乘上余弦窗，解决边界效应，使接近边缘的像素值接近于零
        FeaturesMap = FeaturesMap * self.win

        return FeaturesMap


    def track(self, search_region, img):
        """
         Search the region and find the most similar area of the current template model.

         Args:
             search_region: [ix, iy, w, h] of the search region (which is twice larger than selected bounding box)
             img: A 3-dimension array [w, h, 3]  of an RGB image
                  or a 2-dimension array [w, h] of a gray image
                  of the next frame
         Return:
             ind_x, ind_y: the local coordinates of the found image patch, which has the min measurement.
             ind_s: the index of the size pool which which has the min measurement.

         """
        # 记录最大值、最大值对应偏移位置、最大值对应模型框大小变化率
        max_sim = 0
        local = []
        ind_s = 1.0

        for s in self._scale_pool: # iteration over scale
            search_region_s = list(search_region)
            search_region_s[2] = search_region[2] * s
            search_region_s[3] = search_region[3] * s
            search_region_s = list(map(float, search_region_s))
            s_patch = self.getFeatures(img, search_region_s, self._tmpl_sz)

            # Assignment 1 hints, replace the sliding window
            # Todo
            #  1. convert input search region feature to Fourier domain F
            #  2. use piece-wise dot in Fourier domain to get new searching area patch G' = dot(F,H^*)
            #  3. inverse G' from Fourier domain to g, where every pixel is the measurement.
            #  4. calculate the delta motion of the object from g
            #  5. return the delta/local-coordinates

            # 通过响应输出g来确定位置，通过频域计算出G，再进行IFFT求出g，g矩阵的最大值就是目标的位置
            F = np.fft.fft2(s_patch, axes=(0, 1))
            G = F * self._tmpl  #G' = dot(F,H^*)
            g = np.fft.ifft2(G, axes=(0, 1)).real
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(g) #矩阵的最小值，最大值，以及相应的坐标

            # 找出最佳位置
            if max_val > max_sim:
                max_sim = max_val
                local = max_loc
                ind_s = s

        return local[0], local[1], ind_s

    def update_model(self, x, train_interp_factor):
        """
        Update template model by weighted interpolation.
        Args:
            x: new model in the next frame, that is, feature of the tracked roi
            train_interp_factor: interpolation factor for updating the template model

        # Assignment 1 hints,
        Todo: update the _tmpl as H* = sum(dot(G_i, F_i^*)) / sum(dot(F_i, F_i^*) + e)
            1. Calculate the F_i^*, that is convert the x to the Fourier domain
            2. Calculate the H*
            3. Weighted update the template model
        """
        # x是下一帧里roi的feature
        F = np.fft.fft2(x, axes=(0, 1))
        H = self.G * np.conjugate(F) / (F * np.conjugate(F) + self.e)
        self._tmpl = train_interp_factor * H +(1 - train_interp_factor) * self._tmpl



    def createGaussian(self):  # 构建响应图，只在初始化用
        # 构造高斯矩阵G
        x,y = np.meshgrid(np.arange(-self._tmpl_sz[0]/2, self._tmpl_sz[0]/2), np.arange(-self._tmpl_sz[1]/2,self._tmpl_sz[1]/2)) #生成网格点坐标矩阵

        sigma = 3  #矩阵的方差
        dist = -0.5/(sigma*sigma)*(np.square(x) + np.square(y))  # 利用高斯函数产生的值作为高斯响应
        gaussian = np.exp(dist)
        gaussian = (gaussian-gaussian.min())/(gaussian.max()-gaussian.min()) #归一化

        return np.fft.fft2(gaussian)

    def init(self, roi, image):
        """
        Initialize tracker:
        - Initialize the roi and to the bounding box selected by the usr.
        - initialize the template model to the feature of the roi.

        Args:
            roi: [ix, iy, w, h] roi of the selected bounding box
            image: A 3-dimension array [w, h, 3]  of an RGB image
                   or a 2-dimension array [w, h] of a gray image

        # Assignment 1 hints,
        Todo: Initialize the _tmpl as H* = sum(dot(G_i, F_i^*)) / sum(dot(F_i, F_i^*) + e)
            1. G: construct a initial gaussian matrix, which is the label of every position in the search range
            2. F: get the feature of search region and convert it to Fourier domain
            3. calculate and update the template model by G and F as the formula
        """

        self._roi = list(map(int, roi))
        assert (roi[2] > 0 and roi[3] > 0)

        self.G = self.createGaussian()

        # 设定搜索范围，宽和高都为原面积的两倍
        search_roi = [self._roi[0] - self._roi[2] / 2, self._roi[1] - self._roi[3] / 2, self._roi[2] * 2,
                       self._roi[3] * 2]

        # 计算初始模型
        f = self.getFeatures(image, search_roi, self._tmpl_sz)
        F = np.fft.fft2(f, axes=(0, 1))
        self._tmpl = self.G * np.conjugate(F) / (F * np.conjugate(F) + self.e)




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
        search_rect = [cx - self._roi[2], cy - self._roi[3], self._roi[2] * 2, self._roi[3] * 2]

        # the delta in search region
        # 得到目标最左上角的坐标变化量
        loc_pos = self.track(search_rect, image)
        print(loc_pos)

        # delta_x and delta_y we want to estimate
        delta = (np.array(loc_pos[:2]) - self._tmpl_sz / 2)
        # 因为search_roi是扩展了两倍大小的，并且计算feature时做了resize，因此计算相对_roi的位置要减去_tmpl_sz/2

        # scale between the search_roi and our template
        scale = loc_pos[2] * np.array(search_rect[2:]).astype(float) / (np.array(self._tmpl_sz) * 2)
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

        #下一次的搜索范围
        search_roi = [self._roi[0] - self._roi[2] / 2, self._roi[1] - self._roi[3] / 2, self._roi[2] * 2,
                       self._roi[3] * 2]
        # update the template
        x = self.getFeatures(image, search_roi, self._tmpl_sz)

        self.update_model(x,self._interp_factor)

        return self._roi
