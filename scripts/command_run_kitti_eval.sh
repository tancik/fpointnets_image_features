#!/bin/bash

train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ $1 > $1/log.txt
