import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import torch


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def draw_conf_distribution(tp, conf, pred_cls, target_cls, names, save_dir):
    # i = np.argsort(-conf)
    # tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    conf = conf[tp[:, 0]]
    pred_cls = pred_cls[tp[:, 0]]
    # print(conf.shape)
    # print(pred_cls.shape)
    # Find unique classes
    unique_classes = np.unique(target_cls)
    conf_list = []
    for i, c in enumerate(unique_classes):
        conf_list.append(conf[pred_cls == c])
    # for j in conf_list:
    #     print(j.shape)
    sqrt_c = math.sqrt(len(conf_list))
    h = math.floor(sqrt_c)
    w = math.ceil(sqrt_c)
    if h * w < len(conf_list) or h * w == 1:
        h += 1
    # h = math.floor(sqrt_c) + 1
    # w = math.floor(sqrt_c) + 1
    fig, ax = plt.subplots(h, w, figsize=(12, 6))
    ax = ax.ravel()
    for i in range(len(conf_list)):
        ax[i].scatter(conf_list[i], range(len(conf_list[i])), c=hist2d(conf_list[i], conf_list[i], 90), cmap='jet')
        ax[i].set_title(names[int(unique_classes[i])])
        ax[i].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / 'conf_distribution.png', dpi=200)
    # plt.show()


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def x1y1wh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


if __name__ == '__main__':
    targets = json.load(open('data/zw810/yolo_annotation/instances_val.json'))  # dict
    results = json.load(open('best_zw810-s_results.json'))  # list
    # print(len(results))
    # results对应targets['annotations']
    print(results[0].keys())
    # print(len(targets['annotations']))
    print(targets['annotations'][0].keys())
    print(len(targets['images']))
    results_list = []
    targets_list = []
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    for i in range(len(targets['images'])):
        result_i = [r for r in results if r['image_id'] == i + 1]
        target_i = [t for t in targets['annotations'] if t['image_id'] == i + 1]
        results_list.append(result_i)
        targets_list.append(target_i)
        print(result_i)
        print(target_i)

    stats = []
    for r, t in zip(results_list, targets_list):
        r_boxes = x1y1wh2xyxy(torch.tensor([i['bbox'] for i in r], dtype=torch.float32))
        r_scores = torch.tensor([i['score'] for i in r], dtype=torch.float32)
        r_preds = torch.tensor([i['category_id'] for i in r])
        r_cls = torch.tensor([i['category_id'] - 1 for i in r])
        # print(r_boxes.shape, r_scores.shape, r_preds.shape)
        t_boxes = x1y1wh2xyxy(torch.tensor([i['bbox'] for i in t], dtype=torch.float32))
        t_cls = torch.tensor([i['category_id'] - 1 for i in t])
        # print(t_boxes.shape, t_cls.shape)

        # print(ious.shape)
        correct = torch.zeros(r_preds.shape[0], niou, dtype=torch.bool)
        detected = []
        nl = t_boxes.shape[0]
        for cls in torch.unique(t_cls):
            ti = (cls == t_cls).nonzero(as_tuple=False).view(-1)  # prediction indices
            pi = (cls == r_cls).nonzero(as_tuple=False).view(-1)
            if pi.shape[0]:
                ious, i = box_iou(r_boxes[pi], t_boxes[ti]).max(1)
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[i[j]]  # detected target
                    if d not in detected:
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == nl:  # all targets already located in image
                            break
        stats.append((correct, r_scores, r_cls, t_cls))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    names = ['person', 'car', 'cloth', 'bird', 'flower', 'tie', 'hand', 'smoke', 'phone', 'head',
        'paper' ]
    draw_conf_distribution(*stats, names, save_dir='./')
