import os
import random
import pandas as pd


def list2str(_list):
    res = ""
    for l in _list:
        res += l + " "
    return res
# src_dir = "./0707ObjectDection3D/liver_image_T2"
# file_list = os.listdir(src_dir)

# length = len(file_list)
# fold_length = length // 5
# random.shuffle(file_list)
# index_df = pd.DataFrame(columns=["index"])


# for i in range(5):
#     if i == 4:
#         index_df.loc[str(i), "index"] = list2str(file_list[i * fold_length: length])
#     else:
#         index_df.loc[str(i), "index"] = list2str(file_list[i * fold_length: (i + 1) * fold_length])
src_dir = "data\\resection_V\\images"
file_list = os.listdir(src_dir)

# ex_files = ["117_V.nii.gz", "150_V.nii.gz", "219_V.nii.gz", "22_V.nii.gz", "16_V.nii.gz", "24_V.nii.gz", "176_V.nii.gz", "158_V.nii.gz", "13_V.nii.gz", "128_V.nii.gz",
#             "25_V.nii.gz", "238_V.nii.gz", "141_V.nii.gz", "123_V.nii.gz", "236_V.nii.gz", "116_V.nii.gz", "119_V.nii.gz", "239_V.nii.gz", "184_V.nii.gz", "164_V.nii.gz",
#             "185_V.nii.gz", "1_V.nii.gz", "14_V.nii.gz",  "4_V.nii.gz", "11_V.nii.gz", "174.nii.gz"]

# for _file in ex_files:
#     print(_file)
#     file_list.remove(_file)
# print(len(ex_files))
length = len(file_list)
print(length)
fold_length = length // 5
remain_length = length % 5
remain = 0
random.shuffle(file_list)
index_df = pd.DataFrame(columns=["index"])

for i in range(5):
    start = i * fold_length + remain
    if remain < remain_length:
        end = (i+1) * fold_length + remain + 1
        remain += 1
    else:
        end = (i+1) * fold_length + remain
    index_df.loc[str(i), "index"] = list2str(file_list[start: end])
    print(start, end)
index_df.to_csv("./data/index_V.csv")
