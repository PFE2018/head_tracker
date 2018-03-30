
import numpy as np


def shape_to_array(shape):
    # initialize the list (x,y) coordinates
    coordinates = np.zeros((68,2), dtype='float')
    # loop over the 68 facial landmarks and convert them to a 2 tuple of (x,y) coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)
    
    # return the coordinates list of landamarks
    coordinates = coordinates.astype('float32')
    return coordinates    


def rect_to_boundingbox(rect):
    # convert the rectangluar region given by dlib and convert it to the bounding box format (x,y,w,h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h