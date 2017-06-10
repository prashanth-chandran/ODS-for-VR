#!/usr/bin/env python

'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    #cam = video.create_capture(fn)
    #ret, prev = cam.read()
    path0 = "record0/cam0/"
    path1 = "record0/cam1/"
    pathcam0 = path0 + "0000000"
    pathcam1 = path1 + "0000000"
    temp = cv2.imread(pathcam0 + "000" + ".png")

    mylist = os.listdir(path0)  # dir is your directory path
    pics = len(mylist)
    height, width, layers = temp.shape
    video = cv2.VideoWriter('stereoCam6fps.mp4', 0x00000021, 6, (2 * width, height + 320))
    for index in range(0, pics):
        if index < 10:
            left = cv2.imread(pathcam0 + "00" + str(index) + ".png")
            right = cv2.imread(pathcam1 + "00" + str(index) + ".png")
        elif index < 100:
            left = cv2.imread(pathcam0 + "0" + str(index) + ".png")
            right = cv2.imread(pathcam1 + "0" + str(index) + ".png")
        else:
            left = cv2.imread(pathcam0 + str(index) + ".png")
            right = cv2.imread(pathcam1 + str(index) + ".png")

        leftgray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rightgray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(leftgray, rightgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        row1 = np.hstack((cv2.resize(left, (500, 320)), np.zeros((320, 1, 3), np.uint8)))
        row1 = np.hstack((row1, cv2.resize(draw_flow(rightgray, flow), (500, 320))))
        row1 = np.hstack((row1, np.zeros((320, 1, 3), np.uint8)))
        row1 = np.hstack((row1, cv2.resize(right, (500, 320))))
        row1 = np.hstack((row1, np.zeros((320, 1, 3), np.uint8)))
        row1 = np.hstack((np.zeros((320, 1, 3), np.uint8), row1))
        row2 = np.hstack((draw_hsv(flow), warp_flow(left.copy(), flow)))
        mrg = np.vstack((row1, row2))
        cv2.imshow('Optical Flow Analysis', mrg)

        video.write(mrg)
        ch = cv2.waitKey(5)

cv2.destroyAllWindows()
video.release()