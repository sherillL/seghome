import cv2
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale==1 else 1
    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label

class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255) 
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val=mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple,list)):
            self.mean_val = [int(x* factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)

    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        th, tw = self.output_size
        h, w, _ = img.shape
        assert th <= h and tw <= w, "output size is bigger than image size"
        x = int(round(w - tw) / 2.0)
        y = int(round(h - th) / 2.0)
        return x, y

    def __call__(self, img, label=None):
        x, y = self._get_params(img)
        th, tw = self.output_size
        if label is not None:
            return img[y:y + th, x:x + tw], label[y:y + th, x:x + tw]
        else:
            return img[y:y + th, x:x + tw]


def resize(img, size, interpolation=1):
    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation=interpolation)

# TODO
class Resize(object):
    def __init__(self, size, interpolation=1):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img, label=None):
        img = resize(img, self.size, self.interpolation)
        if label is not None:
            label = resize(label, self.size, self.interpolation)
        if label is None:
            return img
        else:
            return img, label

# TODO
class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if np.random.random() < self.prob:
            if np.random.random() < self.prob:
                img = cv2.flip(img, flipCode=0)
                if label is not None:
                    label = cv2.flip(label, flipCode=0)
                    return img, label
                return img
            else:
                img = cv2.flip(img, flipCode=1)
                if label is not None:
                    label = cv2.flip(label, flipCode=1)
                    return img, label
                return img
        if label is not None:
            return img, label
        else:
            return img

# TODO
class RandomCrop(object):
    def __init__(self, crop_size, im_padding_value=[127.5, 127.5, 127.5], label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError('when crop_size is list or tuple, it should include 2 elements')
        elif not isinstance(crop_size, int):
            raise TypeError('Type of crop_size is invalid. Must be Integer or lisst or tuple')
            
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return im
            else:
                return im, label
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                img_channel = im.shape[2]
                import copy
                orig_im = copy.deepcopy(im)
                im = np.zeros((img_height + pad_height, img_width + pad_width,
                               img_channel)).astype(orig_im.dtype)
                for i in range(img_channel):
                    im[:, :, i] = np.pad(
                        orig_im[:, :, i],
                        pad_width=((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=(self.im_padding_value[i],
                                         self.im_padding_value[i]))

                if label is not None:
                    label = np.pad(label,
                                   pad_width=((0, pad_height), (0, pad_width)),
                                   mode='constant',
                                   constant_values=(self.label_padding_value,
                                                    self.label_padding_value))

                img_height = im.shape[0]
                img_width = im.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(w_off + crop_width
                                                            ), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        if label is None:
            return im
        else:
            return im, label

# TODO
class Scale(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, img, label=None):

        if label is not None:
            return img, label
        else:
            return img

# TODO
class RandomScale(object):
    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, label=None):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    label = label[h1:(h1 + dh), w1:(w1 + dw)]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    label = cv2.resize(
                        label, (img_width, img_height),
                        interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return im
        else:
            return im, label



def main():
    image = cv2.imread('/home/aistudio/work/dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread('/home/aistudio/work/dummy_data/GroundTruth_trainval_png/2008_000064.png')

    # TODO: crop_size
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop
    crop_size = 100
    transform = Compose([RandomScale(), RandomFlip(), Pad(size=5), RandomCrop(crop_size=crop_size)])

    for i in range(10):
        ti, tl = transform(image, label)
        cv2.imwrite("image"+str(i)+".png", ti)
        cv2.imwrite("label"+str(i)+".png", tl)

if __name__ == "__main__":
    main()
