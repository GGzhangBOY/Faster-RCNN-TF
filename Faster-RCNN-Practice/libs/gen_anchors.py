import numpy as np


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(4, 8)):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = ratio_structs(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])

    return anchors

def ratio_structs(anchor, ratios):
    w, h, x_ctr, y_ctr = _mkPos(anchor)
    size = w * h
    size_ratios = size/ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws*ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    
    return anchors

def _mkPos(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1 
    x_ctr = anchor[0] + (w-1) *0.5
    y_ctr = anchor[1] + (h-1) *0.5

    return w,h,x_ctr,y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))#返回左上，右下的坐标
    return anchors

def _scale_enum(anchors,scales):
    w,h,x_ctr,y_ctr = _mkPos(anchors)
    ws = w*scales
    hs = h*scales
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

if __name__ == '__main__':
    print(generate_anchors())
