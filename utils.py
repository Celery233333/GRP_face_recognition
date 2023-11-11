import math
import cv2
import numpy as np


# calculate the scale of each scaling of the original input image
def calculateScales(img):
    pr_scale = 1.0
    h,w,_ = img.shape
    
    # make a fix image size 500
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0/max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    # create scales of the image pyramid
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)

    # prevent the width and height of the image less than 12
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


# change rectangle to square
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles


# nonmaximal inhibition
def NMS(rectangles,threshold):
    # check the empty
    if len(rectangles)==0:
        return rectangles
    
    # get the positions of rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]

    # calculate the areas
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []

    # get the maximum one
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


# process result from pnet 
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    # calculate the length between feature points
    stride = 0
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1)

    # obtain the coordinates of the feature points that meet the threshold and expand dimension
    (y,x) = np.where(cls_prob >= threshold)
    score = np.expand_dims(cls_prob[y, x], -1)

    # the coordinates of the corresponding feature points are converted to the coordinates of the prior box on the original graph
    boundingbox = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis = -1)
    top_left = np.fix(stride * boundingbox + 0)
    bottom_right = np.fix(stride * boundingbox + 11)

    # get the corresponding rough prediction box
    boundingbox = np.concatenate((top_left,bottom_right), axis = -1)
    boundingbox = (boundingbox + roi[y, x] * 12.0) * scale
    
    # stack the prediction boxes and scores and convert them to squares
    rectangles = np.concatenate((boundingbox, score), axis = -1)
    rectangles = rect2square(rectangles)
    
    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)
    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
    return rectangles
    

# process result from rnet
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    # use scores for screening
    pick = cls_prob[:, 1] >= threshold
    score  = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    roi = roi[pick, :]

    # expand dimensions
    w   = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h   = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)

    # use result from rnet to adjust the rough prediction box
    rectangles[:, [0,2]]  = rectangles[:, [0,2]] + roi[:, [0,2]] * w
    rectangles[:, [1,3]]  = rectangles[:, [1,3]] + roi[:, [1,3]] * h

    # stack the prediction boxes and scores and convert them to squares
    rectangles = np.concatenate((rectangles,score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)
    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
    return np.array(NMS(rectangles, 0.7))


# process the result from onet
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    # use scores for screening
    pick = cls_prob[:, 1] >= threshold
    score  = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    pts = pts[pick, :]
    roi = roi[pick, :]

    # expand dimensions
    w   = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h   = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)

    # use result from rnet to adjust the rough prediction box
    face_marks = np.zeros_like(pts)
    face_marks[:, [0,2,4,6,8]] = w * pts[:, [0,1,2,3,4]] + rectangles[:, 0:1]
    face_marks[:, [1,3,5,7,9]] = h * pts[:, [5,6,7,8,9]] + rectangles[:, 1:2]
    rectangles[:, [0,2]]  = rectangles[:, [0,2]] + roi[:, [0,2]] * w
    rectangles[:, [1,3]]  = rectangles[:, [1,3]] + roi[:, [1,3]] * w

    # stack the prediction boxes and scores and convert them to squares
    rectangles = np.concatenate((rectangles,score,face_marks),axis=-1)
    print(score)
    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)
    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
    return np.array(NMS(rectangles,0.3))

# pre-procession
def pre_process(x):
    # check dimension
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    # normalization
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

# 12 normalization
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


# calculate 128 eigenvalues
def calc_128_vec(model,img):
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre,[128])
    return pre


# calculate face distance
def face_distance(face_encodings, face_to_compare):
    # check empty situation
    if len(face_encodings) == 0:
        return np.empty((0))

    # calculate the cosine distance
    a_norm = np.linalg.norm(face_encodings)
    b_norm = np.linalg.norm(face_to_compare)
    similarity = np.dot(face_encodings, face_to_compare)/(a_norm * b_norm)
    return similarity


# process and save photo
def face_process(rectangle,img):
    featrue = [round(x) for x in rectangle]
    face = img[featrue[1]:featrue[3],featrue[0]:featrue[2]]
    
    # get the both sides of the triangle
    x = featrue[7] - featrue[5]
    y = featrue[8] - featrue[6]
   
    # get the radian of it
    angle = math.atan(y/x)*180/math.pi
   
    # get the center of the image
    h, w = face.shape[:2]
    center = (w//2,h//2)
   
    # rotate the image
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_image = cv2.warpAffine(face,RotationMatrix,(w,h))
    new_image = letterbox(new_image, (160,160))
    return new_image

# resize the image and fill black bars in non-face areas
def letterbox(image_src,dst_size):
    # get the size and target size
    src_h,src_w = image_src.shape[:2]
    dst_h,dst_w = dst_size 

    # calculate the proportion that needs to be modified
    h = dst_w * (float(src_h)/src_w)
    w = dst_h * (float(src_w)/src_h)
   
    h = int(h)
    w = int(w)
   
   # source image may be smaller or larger than the target
    if h <= dst_h:
        image_dst = cv2.resize(image_src,(dst_w,int(h)))
    else:
        image_dst = cv2.resize(image_src,(int(w),dst_h))
   
    h_,w_ = image_dst.shape[:2]
    
    # calculate the new position
    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_+1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_+1) / 2)
        
    value = [0,0,0]
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
    
    return image_dst