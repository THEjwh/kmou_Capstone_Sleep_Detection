import pandas as pd
import glob
import numpy as np
import shutil

folders = glob.glob('./dataset/HeadPoseImageDatabase/*', recursive=True)
test = glob.glob(folders[0] + '/*.jpg')
count = 0
for i in folders:
    images = glob.glob(i + '/*.jpg')

    targetstr = []
    targetstrx = ['-15', '+0' ,'+15','+30', '+60']
    targetstry = ['-90', '-75', '-60', '-45', '-30', '-15', '+0', '+15' , '+30' , '+45' , '+60', '+75', '+90']
    for s in targetstrx:
        for ss in targetstry:
            targetstr.append(s+ss)
    

    for j in images:
        for zxc in targetstr:
            if zxc in j:
                shutil.move(j, "./dataset/headpose/stand/" + 'stand_'+ str(count) + '.jpg')
                count += 1
                break

