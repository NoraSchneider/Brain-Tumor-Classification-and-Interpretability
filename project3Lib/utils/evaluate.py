import numpy as np


def iou(prediction, target):
    prediction = np.array(prediction, dtype = bool)
    target = np.array(target, dtype=bool)
    assert target.shape == prediction.shape
    overlap = prediction*target
    union = prediction + target
    return 1.0 if float(union.sum()) == 0 else overlap.sum()/float(union.sum())


def evaluate_interpretability(prediction, true_mask, number_pixels):
    assert prediction.shape == true_mask.shape
    threshold = np.copy(prediction).flatten()
    threshold.sort()
    threshold = threshold[len(np.copy(prediction).flatten())-1-number_pixels]
    helper = np.copy(prediction)
    threshold = 0 if threshold<0 else threshold
    helper[(prediction > threshold)] = 1
    helper[prediction <= threshold] = 0
    return iou(helper, true_mask)