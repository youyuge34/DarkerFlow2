#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-05-21 23:42
# @Author  : YouSheng
import json
import os
from time import time as timer
import numpy as np
import cv2
from darkflow.net.build import TFNet
import math
import time

def postprocess(self, net_out, im, save=True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.framework.findboxes(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im
    h, w, _ = imgcv.shape

    resultsForJSON = []

    # 预处理boxes，因为同一人物不可能同时出现多个，因此只保留confidence最大的人物box
    # 记录最大概率
    pro_max = [-1, -1, -1]
    # 记录3人各自最大confidence的index
    pro_max_index = [0, 0, 0]
    boxes_post = []
    for i, b in enumerate(boxes):
        boxResults = self.framework.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        _, _, _, _, _, max_indx, confidence = boxResults
        if confidence > pro_max[max_indx]:
            pro_max[max_indx] = confidence
            pro_max_index[max_indx] = i

    for i in range(len(pro_max_index)):
        if len(boxes) > pro_max_index[i]:
            boxes_post.append(boxes[pro_max_index[i]])

    if len(boxes_post) != 0:
        for b in boxes_post:
            boxResults = self.framework.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, mess, max_indx, confidence = boxResults

            thick = int((h + w) // 300)
            if self.FLAGS.json:
                resultsForJSON.append(
                    {"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
                     "bottomright": {"x": right, "y": bot}})
                continue

            cv2.rectangle(imgcv,
                          (left, top), (right, bot),
                          colors[max_indx], thick)
            cv2.putText(imgcv, mess + ' ' + str(float('%.2f' % confidence)), (left, top - 12),
                        0, 1e-3 * h, colors[max_indx], thick // 3)

            # 给雪菜打码
            if 'Setsuna' == mess:
                cv2.line(imgcv,
                          (left, top), (right, bot),
                         (0,0,255), thick*2)
                cv2.line(imgcv,
                          (right, top), (left, bot),
                         (0,0,255), thick*2)


    if not save: return imgcv

    outfolder = os.path.join(self.FLAGS.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))
    if self.FLAGS.json:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)
        return

    # print('saving:', img_name)
    cv2.imwrite(img_name, imgcv)


def camera(self, output):
    file = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo
    
    if file == 'camera':
        file = 0
    else:
        assert os.path.isfile(file), \
        'file {} does not exist'.format(file)
        
    camera = cv2.VideoCapture(file)
    
    if file == 0:
        self.say('Press [ESC] to quit demo')
        
    assert camera.isOpened(), \
    'Cannot capture source'
    
    if file == 0:#camera window
        cv2.namedWindow('', 0)
        _, frame = camera.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('', width, height)
    else:
        _, frame = camera.read()
        height, width, _ = frame.shape

    if SaveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if file == 0:#camera window
          fps = 1 / self._get_fps(frame)
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(
            output, fourcc, fps, (width, height))

    # buffers for demo in batch
    buffer_inp = list()
    buffer_pre = list()
    
    elapsed = int()
    start = timer()
    self.say('Press [ESC] to quit demo')
    # Loop through frames
    while camera.isOpened():
        elapsed += 1
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video')
            break
        preprocessed = self.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)
        
        # Only process and imshow when queue is full
        if elapsed % self.FLAGS.queue == 0:
            feed_dict = {self.inp: buffer_pre}
            net_out = self.sess.run(self.out, feed_dict)
            for img, single_out in zip(buffer_inp, net_out):
                postprocessed = postprocess(tfnet,
                    single_out, img, False)
                if SaveVideo:
                    videoWriter.write(postprocessed)
                if file == 0: #camera window
                    cv2.imshow('', postprocessed)
            # Clear Buffers
            buffer_inp = list()
            buffer_pre = list()

        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        if file == 0: #camera window
            choice = cv2.waitKey(1)
            if choice == 27: break

    sys.stdout.write('\n')
    if SaveVideo:
        videoWriter.release()
    camera.release()
    if file == 0: #camera window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    options = {
        'model': 'cfg/tiny-yolo-voc-wa1.cfg',
        'load': 2500,
        'threshold': 0.05,
        'gpu': 0.8,
        'demo': 'sample_video/WA_CM02.mp4',
        'saveVideo': True,
    }
    output = 'bin/'+options['demo'].split('/')[1].split('.')[0]+'_'+options['load']+'_'+options['threshold']+'_'+options['model'].split('/')[1].split('.')[0]+'.avi'
    tfnet = TFNet(options)
    camera(tfnet, output)


