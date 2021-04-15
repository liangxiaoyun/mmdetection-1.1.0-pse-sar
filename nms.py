import numpy as np

def nms(bbox, thres=0.5):
    '''
    bbox:np.array([[xmin,ymin,xmax,ymax,score],...])
    :param bbox:
    :return:
    '''
    xmin = bbox[:, 0]
    xmax = bbox[:, 1]
    ymin = bbox[:, 2]
    ymax = bbox[:, 3]
    score = bbox[:,4]
    area = (ymax-ymin+1) * (xmax-xmin+1)

    index = score.argsort()[::-1]
    keep = []

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(xmin[i], xmin[index[1:]])
        x22 = np.minimum(xmax[i], xmax[index[1:]])
        y11 = np.maximum(ymin[i], ymin[index[1:]])
        y22 = np.minimum(ymax[i], ymax[index[1:]])

        w = np.maximum(x22 - x11 + 1, 0)
        h = np.maximum(y22 - y11 + 1, 0)

        overlap = w * h
        ious = overlap / (area[i] + area[index[1:]] - overlap)
        idx = np.where(ious < thres)[0]
        index = index[idx+1]
    return bbox[keep]

def mutil_class_nms(det):
    result = []
    for each in range(2):
        bbox = det[np.where(det[:, 5:7].argsort()[:,-1] == each)[0].tolist(), :]
        nms_bbox = nms(bbox, thres=0.5)
        result.append(nms_bbox)
    return result

if __name__ == '__main__':
    # bbox = np.array([[0,1,2,3,0.1],[0,1,2,3,0.1],[0,1,2,3,0.1],[0,1,2,3,0.1]])
    # nms_bbox = nms(bbox, thres=0.5)
    det = np.array([np.array([0,1,2,3,0.1,0.4,0.6]),np.array([0,1,2,3,0.1,0.4,0.6]),np.array([0,1,2,3,0.1,0.4,0.6]),np.array([0,1,2,3,0.1,0.4,0.6])])
    result = mutil_class_nms(det)
    print(result)

