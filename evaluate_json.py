import logging
import itertools
import logging
import os.path as osp
import torch.distributed as dist

from terminaltables import AsciiTable
import itertools

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def evaluate(results_json, targets_json):
    class_name = (
        'person', 'car', 'cloth', 'bird', 'flower', 'tie', 'hand', 'smoke', 'phone', 'head', 'paper')
    # _valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # cat_ids = {v: i for i, v in enumerate(_valid_ids)}
    cocoGt = COCO(targets_json)
    cat_ids = cocoGt.getCatIds(catNms=class_name)
    imgIds = cocoGt.getImgIds()
    coco_dets = cocoGt.loadRes(results_json)
    coco_eval = COCOeval(cocoGt, coco_dets, "bbox")
    # print(coco_eval.params.maxDets)
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    precisions = coco_eval.eval['precision']
    # precision: (iou, recall, cls, area range, max dets)
    assert len(cat_ids) == precisions.shape[2]

    results_per_category = []
    for idx, catId in enumerate(cat_ids):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = cocoGt.loadCats(catId)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        results_per_category.append(
            (f'{nm["name"]}', f'{float(ap):0.3f}'))

    num_columns = min(6, len(results_per_category) * 2)
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['category', 'AP'] * (num_columns // 2)
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=None)
    map, map50 = coco_eval.stats[:2]
    return map


if __name__ == '__main__':
    evaluate('your_results.json', 'your_instances.json')
