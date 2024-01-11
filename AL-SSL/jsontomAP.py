import os
import cv2
import json
from tqdm import tqdm
import shutil
import re

currentDirPath = os.getcwd()
rawDirPath = currentDirPath + "/eval/coco2014"
rawPath_list = os.listdir(rawDirPath) # ['{}.json']
rawPath_list.sort()
# print("rawPath_list:", rawPath_list)
rawFullPath_list = [
            rawDirPath + '/' + file_name for file_name in rawPath_list] 
            # '/home/yoonk/workspace/active-learning/AL-SSL/eval/val2014/result-500.json
# print("rawFullPath_list:", rawFullPath_list)
score_list = []

for rawFullPath in tqdm(rawFullPath_list):
    # source= cv2.imread("{}".format(img))
    res = re.split('[/ .]+', rawFullPath)
    print("Path:", res[-2])
    with open(rawFullPath, 'r') as f:
        data = json.load(f)
        mAPscore = 0
        for single_data in data:
            mAPscore += single_data['score']
        mAPscore /= len(data)
        score_list.append(mAPscore)
        print("len(data):", len(data))
        print("mAPscore:", mAPscore)
        print("=====================================")
print("score_list:", score_list)

    