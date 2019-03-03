import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training RPN')
    parser.add_argument('folder_1', type=str)
    parser.add_argument('folder_2', type=str)
    parser.add_argument('--axis', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    files_1 = os.listdir(args.folder_1)
    files_2 = os.listdir(args.folder_2)

    files = list(set(files_1) & set(files_2))

    for i, file in enumerate(files):
        im_1 = cv2.imread(args.folder_1 + '/' + file)
        im_2 = cv2.imread(args.folder_2 + '/' + file)
        im = np.concatenate([im_1, im_2], axis=args.axis)

        if args.save_dir is not None:
            cv2.imwrite(args.save_dir + '/im_' + str(i) + '.png', im)
        else:
            cv2.imshow('image', im)
            cv2.waitKey(1000)
        # cv2.destroyAllWindows()
