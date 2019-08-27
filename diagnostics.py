from __future__ import division

import numpy as np
import cv2


def iou_for_semantic_class(target_color_channel, inferenced_color_channel):
    intersection = np.logical_and(target_color_channel, inferenced_color_channel)
    union = np.logical_or(target_color_channel, inferenced_color_channel)
    if np.sum(union) == 0.0:
        return 0.0
    return np.sum(intersection) / np.sum(union)


def acc_for_semantic_class(target_color_channel, inferenced_color_channel):
    true_pos = np.logical_and(target_color_channel, inferenced_color_channel)
    true_neg = np.logical_and(np.logical_not(target_color_channel), np.logical_not(inferenced_color_channel))
    h, w = target_color_channel.shape
    return (np.sum(true_pos) + np.sum(true_neg)) / (h * w)


def acc_rgb(img_1, img_2):
    b = acc_for_semantic_class(img_1[:, :, 0], img_2[:, :, 0])
    g = acc_for_semantic_class(img_1[:, :, 1], img_2[:, :, 1])
    r = acc_for_semantic_class(img_1[:, :, 2], img_2[:, :, 2])
    return r, g, b


def iou_rgb(img_1, img_2):
    b = iou_for_semantic_class(img_1[:, :, 0], img_2[:, :, 0])
    g = iou_for_semantic_class(img_1[:, :, 1], img_2[:, :, 1])
    r = iou_for_semantic_class(img_1[:, :, 2], img_2[:, :, 2])
    return r, g, b


def preprocess_inference(inf, threshold):
    _, b = cv2.threshold(inf[:, :, 0], threshold[0], 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(inf[:, :, 1], threshold[1], 255, cv2.THRESH_BINARY)
    _, r = cv2.threshold(inf[:, :, 2], threshold[2], 255, cv2.THRESH_BINARY)
    inf[:, :, 0] = b
    inf[:, :, 1] = g
    inf[:, :, 2] = r
    return inf


def precision4channel(gt, inf):
    true_pos = np.sum(np.logical_and(gt, inf))
    predicted_cond_pos = np.sum(inf) / 255.0
    if predicted_cond_pos == 0:
        return 0.0
    return true_pos / predicted_cond_pos


def precision_rgb(gt, inf):
    # b, g, r = np.sum(np.logical_and(gt, inf), axis = (0, 1)) / (np.sum(inf, axis = (0, 1)) / 255.0)
    b = precision4channel(gt[:, :, 0], inf[:, :, 0])
    g = precision4channel(gt[:, :, 1], inf[:, :, 1])
    r = precision4channel(gt[:, :, 2], inf[:, :, 2])
    return r, g, b


def recall4channel(gt, inf):
    true_pos = np.sum(np.logical_and(gt, inf))
    cond_pos = np.sum(gt) / 255.0
    if cond_pos == 0:
        return 0.0
    return true_pos / cond_pos


def recall_rgb(gt, inf):
    b = recall4channel(gt[:, :, 0], inf[:, :, 0])
    g = recall4channel(gt[:, :, 1], inf[:, :, 1])
    r = recall4channel(gt[:, :, 2], inf[:, :, 2])
    return r, g, b


def f1score_rgb(gt, inf):
    rr, rg, rb = recall_rgb(gt, inf)
    pr, pg, pb = precision_rgb(gt, inf)
    if pr == 0 and rr == 0:
        f1r = 0.0
    else:
        f1r = 2.0 * pr * rr / (pr + rr)
    if pg == 0 and rg == 0:
        f1g = 0.0
    else:
        f1g = 2.0 * pg * rg / (pg + rg)
    if pb == 0 and rb == 0:
        f1b = 0.0
    else:
        f1b = 2.0 * pb * rb / (pb + rb)
    return f1r, f1g, f1b


def colour_quota_rgb(img):
    h, w, _ = img.shape
    b, g, r = np.sum(img, axis = (0, 1)) / (255.0 * h * w)
    return r, g, b


def calc_best_th(por_3C):
    best_th = np.zeros(3)
    best_por = np.zeros(3)
    for i, vals in enumerate(por_3C):
        for k, (precision, recall) in enumerate(vals):
            por = precision * recall
            if por > best_por[k]:
                best_th[k] = i
                best_por[k] = por
    return best_th, best_por