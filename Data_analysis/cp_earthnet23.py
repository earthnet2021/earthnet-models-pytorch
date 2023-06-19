import shutil
import os
from tqdm import tqdm
import json

file = open("Data_analysis/missing_value_test_from2016.json")
data = json.load(file)

os.mkdir("/scratch/crobin/earthnet2023/earthnet2023_preprocessed/")
os.mkdir("/scratch/crobin/earthnet2023/earthnet2023_preprocessed/train/")

src_path = "/scratch/crobin/earthnet2023/train/"
dst_path = "/scratch/crobin/earthnet2023/earthnet2023_preprocessed/train/"
for i, file in enumerate(tqdm(list(data.keys()))):
    if data[file]['total_missing'] / (len(data[file]["serie_missing"]) * 128 * 128) < 0.15:
        shutil.copy(os.path.join(src_path, file[35:]), dst_path)