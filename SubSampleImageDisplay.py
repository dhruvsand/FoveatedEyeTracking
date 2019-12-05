#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

""" Testing the outer product on an image array """

import numpy as np
from PIL import Image

IMAGE = './Landscape.jpg'
SUB   = 4

class OpenImage:
    """ Image opening context manager """
    def __init__(self, imageName: str ='', NAXIS: int =0) -> None:
        self.imageName = imageName
        self.NAXIS = NAXIS

    def __enter__(self) -> np.ndarray:
        self.image = Image.open(self.imageName)
        if (''.join(self.image.getbands()) == 'RGB'):
            return np.rot90(np.array(self.image).T[self.NAXIS], 3)
        else:
            return np.array(self.image)

    def close(self) -> None:
        self.image.close()

    def __exit__(self, *args) -> None:
        self.close()


class Sampling:
    """ Sampling and subsampling routines """
    SUBBED = False
    CACHE  = []

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.__X, self.__Y, *_ = image.shape

    def subSample(self, pix: int =3) -> np.ndarray:
        """ Subsample array in the context of Dokkum's cosmic ray removal algorithm """
        id = np.ones((pix, pix))
        self.image = np.kron(self.image, id)  # update internal state
        self.SUBBED = True
        self.CACHE.append(pix)
        return self.image

    def blockAverage(self, pix: int =3) -> np.ndarray:
        """ Block average array after it's been subsampled """
        assert self.SUBBED, 'Must subSample before blockAverage'
        # if the former is true, then the following should be as well
        assert not all(map(lambda x: x % pix, self.image.shape)),\
            'image dimensions not cleanly divisible by pix (?)'
        assert pix == self.CACHE[-1], 'Must use last subsample {0}, received {1}'\
            .format(self.CACHE[-1], pix)

        # now block average---takes a few seconds
        avrgd = np.empty((self.__X, self.__Y))
        for i in range(self.__X):
            for j in range(self.__Y):
                Xs = i * pix, i * pix + pix
                Ys = j * pix, j * pix + pix
                avrgd[i][j] = np.mean(self.image[slice(*Xs), slice(*Ys)])

        self.image = avrgd
        self.SUBBED = False
        return self.image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with OpenImage(IMAGE) as im:
        s = Sampling(im)

        fig = plt.figure()
        ax1 = fig.add_subplot(122)
        ax1.imshow(s.subSample(SUB), interpolation='none', cmap='gray')
        ax1.set_title('Subsampled')

        # imgSubSample = s.subSample(SUB)
        # img = Image.fromarray(imgSubSample, 'RGB')
        # img.save('stupid.jpg')
        # img.show()




        ax2 = fig.add_subplot(121)
        ax2.imshow(s.blockAverage(SUB), interpolation='none', cmap='gray')
        ax2.set_title('Block averaged')

        plt.show()