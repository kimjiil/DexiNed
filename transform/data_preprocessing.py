from .transforms import *

class TrainAugmentation_Synth:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std

        # train augment
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            # Expand(self.mean),
            ToPercentCoords(),
            Resize(self.size),
            RandomGaussianBlur(),
            ImgNormalize(self.mean,self.std),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)

class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std

        # train augment
        self.augment = Compose([
            RandomSampleCrop_OCRver(),
            ConvertFromInts(),
            PhotometricDistort(),
            # Expand(self.mean),
            ToPercentCoords(),
            Resize(self.size),
            RandomGaussianBlur(),
            ImgNormalize(self.mean,self.std),
            # lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.mean = mean
        self.size = size
        self.std = std
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            ImgNormalize(self.mean,self.std),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.transform = Compose([
            Resize(size),
            ImgNormalize(self.mean,self.std),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image