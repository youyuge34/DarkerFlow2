#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-05-21 20:50
# @Author  : YouSheng
import json
import os
from time import time as timer
import numpy as np
import cv2
from darkflow.net.build import TFNet
import math
from multiprocessing.pool import ThreadPool
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


if __name__ == "__main__":
    options = {
        'model': 'cfg/tiny-yolo-voc-wa1.cfg',
        'load': 1500,
        'threshold': 0.05,
        'gpu': 0.8,
        'imgdir': 'sample_img\wa',
    }
    pool = ThreadPool()
    tfnet = TFNet(options)

    inp_path = tfnet.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    # all_inps为imgdir文件夹内所有图片文件的名字
    all_inps = [i for i in all_inps if tfnet.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(tfnet.FLAGS.batch, len(all_inps))

    # predict in batches
    # 分为了n_batch个batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]

        # 对batch中每一个图片文件进行多进程操作：
        # 在非training的predict中resize，BGR->RGB,使用cv转化为numpy tensor
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(tfnet.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        # 将一个batch内的所有图片concat起来成一个batch送入网络中
        feed_dict = {tfnet.inp : np.concatenate(inp_feed, 0)}
        tfnet.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = tfnet.sess.run(tfnet.out, feed_dict)
        stop = time.time(); last = stop - start
        tfnet.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        # 将tf的out进行后处理，给图片画框并存储
        tfnet.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            postprocess(tfnet,
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        tfnet.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))