import os

file_base = 'dataset/KITTI/object/training/label_2/'

with open('kitti_object_vis/val.txt', 'rb') as f:
    vals = f.readlines()
vals = [int(val) for val in vals]

e_classes = {}
m_classes = {}
h_classes = {}
for idx in vals:
    filename = os.path.join(file_base, '%06d.txt'%(idx))

    with open(filename, 'r') as f:
        for line in f.readlines():
            data = line.split(' ')
            class_type = data[0]
            trunc = float(data[1])
            occ = int(data[2])
            height = data[8]
            if trunc < .15 and occ == 0:
                if class_type in e_classes:
                    e_classes[class_type] += 1
                else:
                    e_classes[class_type] = 1
            if trunc < .30 and occ <= 1:
                if class_type in m_classes:
                    m_classes[class_type] += 1
                else:
                    m_classes[class_type] = 1
            if trunc < .50 and occ <= 2:
                if class_type in h_classes:
                    h_classes[class_type] += 1
                else:
                    h_classes[class_type] = 1
print(e_classes)
print(m_classes)
print(h_classes)
