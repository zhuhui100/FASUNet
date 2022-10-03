import random
import numpy as np
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
from random import choice
import math

import torch


class PadXY_32(object):

    def __call__(self, image, label):
        delta = 32
        nx, ny = math.ceil(image.shape[-2]/delta) , math.ceil(image.shape[-1]/delta)
        x, y = nx*delta, ny*delta
        new_label = np.zeros([x, y], np.int16)
        

        new_image = np.zeros([3, x, y])
        new_image[0:3, 0:image.shape[-2], 0:image.shape[-1]] = image

        new_label[0:label.shape[-2], 0:label.shape[-1]] = label
        return new_image, new_label
        
        
class MinMaxNormalize(object):

    def __call__(self, image, label):
        def norm(im):
            im = im.astype(np.float32)
            min_v = np.min(im)
            max_v = np.max(im)
            im = (im - min_v)/(max_v - min_v)
            return im
        image = norm(image)

        return [image, label]


class RandomFlip(object):


    def __call__(self, image, label):

        if random.random() < 0.5:
            flip_type = np.random.randint(0, 3) # flip across any 3D axis

            image = np.flip(image, flip_type)
            label = np.flip(label, flip_type)

        return [image, label]


class RandCrop_overlap(object):

    def __call__(self, image, label):

        def get_start(length, size, delta):

            points = list(np.arange(0, length-size+1, delta))
            if (length-size)%delta != 0:  
                points.append(length-size)
            return points

        shape = list(image.shape)
        size = [64, 64, 64]
        delta = [10, 10, 10]
        start = []   
        for i in range(len(shape)):
            points = get_start(shape[i], size[i], delta[i])
            start.append(choice(points))
            # print(points, start[i])

        x, y, z = start[0], start[1], start[2]
        image = image[x: x+size[0], y: y+size[1], z: z+size[2]]
        label = label[x: x+size[0], y: y+size[1], z: z+size[2]]

        return [image, label]


class RandCrop(object):

    def __call__(self, image, label):


        dx, dy, dz = 64, 64, 64
        start = []   
        x0 = np.random.randint(0, image.shape[0]-dx)
        y0 = np.random.randint(0, image.shape[1]-dy)
        z0 = np.random.randint(0, image.shape[2]-dz)
        if x0+dx>=image.shape[0] or y0+dy>=image.shape[1] or \
                z0+dz>=image.shape[2]:
            raise RuntimeError("剪切大小超标")
        image = image[x0: x0+dx, y0: y0+dy, z0: z0+dz]
        label = label[x0: x0+dx, y0: y0+dy, z0: z0+dz]

        return [image, label]


class Normalize(object):

    def __call__(self, image, label):
        image = image/image.max() - 0.5
        return [image, label]


class PadXYZ_8(object):

    def __call__(self, image, label):
        delta = 8
        nx, ny = image.shape[0]//delta +1, image.shape[1]//delta +1
        nz = image.shape[2]//delta +1
        x, y, z = nx*delta, ny*delta, nz*delta
        new_image = np.zeros([x, y, z])
        new_label = np.zeros([x, y, z], np.int16)

        new_image[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image
        new_label[0:label.shape[0], 0:label.shape[1], 0:label.shape[2]] = label
        return new_image, new_label


class PadZ_8(object):

    def __call__(self, image, label):
        delta = 8
        nz = math.ceil(image.shape[2]/delta)
        z = nz*delta
        x, y, z = image.shape[0], image.shape[1], nz*delta
        new_image = np.zeros([x, y, z])
        new_label = np.zeros([x, y, z], np.int16)

        new_image[..., 0:image.shape[2]] = image
        new_label[..., 0:label.shape[2]] = label
        return new_image, new_label


class PadXYZ_16(object):

    def __call__(self, image, label):
        delta = 16
        nx, ny = image.shape[0]//delta +1, image.shape[1]//delta +1
        nz = image.shape[2]//delta +1
        x, y, z = nx*delta, ny*delta, nz*delta
        new_image = np.zeros([x, y, z])
        new_label = np.zeros([x, y, z], np.int16)

        new_image[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image
        new_label[0:label.shape[0], 0:label.shape[1], 0:label.shape[2]] = label
        return new_image, new_label


class ResizeXoY(object):

    def __call__(self, image, label):
        h, w = 128/image.shape[0], 128/image.shape[1]
        new_image = zoom(image, (h, w, 1), order=0)
        new_label = zoom(label, (h, w, 1), order=0)
        return [new_image, new_label]




class RandCutZ(object):

    def __call__(self, image, label):
        sub_depth = 64
        if image.shape != label.shape:
            raise RuntimeError("图片和标签维数不一样")
        else:
            x, y, z = image.shape
            h, w = 128/image.shape[0], 128/image.shape[1]
            if z >= sub_depth:
                z0 = np.random.randint(0, z-sub_depth)
            else:
                raise RuntimeError("子区域高度超标")

            image = np.array(image[:, :, z0:z0+sub_depth])
            label = np.array(label[:, :, z0:z0+sub_depth])

            image = zoom(image, (h, w, 1), order=0)
            label = zoom(label, (h, w, 1), order=0)
        
            return image, label


class RandCutX(object):

    def __call__(self, image, label):
        sub_high = 64
        if image.shape != label.shape:
            raise RuntimeError("image size error")
        else:
            x, y, z = image.shape
            w, d = 128/image.shape[1], 128/image.shape[2]
            if x >= sub_high:
                x0 = np.random.randint(0, x-sub_high)
            else:
                raise RuntimeError("errof")

            image = np.array(image[x0:x0+sub_high, :, :])
            label = np.array(label[x0:x0+sub_high, :, :])

            image = zoom(image, (1, w, d), order=0)
            label = zoom(label, (1, w, d), order=0)
        
            return image, label


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for transform in self.transforms:
            args = transform(*args)
        return args



if __name__ == "__main__":
    x = np.random.random((96, 96, 100))
    # y, z = MinMaxNormalize()(x, x)
    y, z = RandCrop()(x, x)

    print(y.max(), y.shape, z.shape)
    # print([3, 6, 8]*3)