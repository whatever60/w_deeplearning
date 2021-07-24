import os

import cv2 as cv
import numpy as np
from tqdm.auto import tqdm

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector


def process(in_dir, filename, out_dir, output_size):
    img = cv.imread(f'{in_dir}/{filename}')
    _, facial5points = detector.detect_faces(img)
    facial5points = np.reshape(facial5points[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(img, facial5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    cv.imwrite(f'{out_dir}/{filename}', dst_img)
    # img = cv.resize(img, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)


if __name__ == "__main__":
    from rich.traceback import install
    install()

    detector = MtcnnDetector()
    data_dir = '/home/tiankang/wusuowei/data/face/casia_webface'
    in_dir = f'{data_dir}/CASIA-WebFace'
    out_dir = f'{data_dir}/aligned'
    f = open(f'{data_dir}/unaligned.txt', 'w')
    for name in tqdm(os.listdir(in_dir)):
        os.makedirs(f'{out_dir}/{name}')
        for image in os.listdir(f'{in_dir}/{name}'):
            filename = f'{name}/{image}'
            try:
                process(in_dir, filename, out_dir, output_size=(112, 112))
            except (IndexError, ValueError):
                print(filename, file=f)
