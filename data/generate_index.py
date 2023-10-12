import os
import random
import pandas as pd


def list2str(_list):
    res = ""
    for l in _list:
        res += l + " "
    return res

src_dir = "./data/resection_V/images"
file_list = os.listdir(src_dir)

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
