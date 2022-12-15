import numpy as np
import cv2

def cos_win(w,h):
    win_col = np.hanning(w)  # 通过hanning窗生成余弦窗
    win_row = np.hanning(h)
    mask_col, mask_row = np.meshgrid(win_col, win_row)  # 生成网格点坐标矩阵
    return mask_col*mask_row

img=cv2.imread('../33.png')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("img",gray)
cv2.waitKey(0)
f=np.fft.fft2(gray, axes=(0, 1))


win = cos_win(img.shape[1],img.shape[0])
cv2.imshow('win',win)
cv2.waitKey(0)

m=win*gray
#m=np.fft.ifft2(m, axes=(0, 1)).real
cv2.imshow('win',m)
cv2.waitKey(0)
