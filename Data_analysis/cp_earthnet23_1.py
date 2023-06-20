import shutil
import os
from tqdm import tqdm
import json

file = open("missing_value_train_from2016.json")
data = json.load(file)

# os.mkdir("/scratch/crobin/earthnet2023_preprocessed/")
# os.mkdir("/scratch/crobin/earthnet2023_preprocessed/train/")

src_path = "/Net/Groups/BGI/work_1/scratch/s3/earthnet/earthnet2023/train/"
dst_path = "/scratch/crobin/earthnet2023_preprocessed/train/"
for i, file in enumerate(tqdm(list(data.keys()))):
    if data[file]['total_missing'] / (len(data[file]["serie_missing"]) * 128 * 128) < 0.15:
        print(os.path.join(src_path, file[35:]), os.path.join(dst_path, file[35:-12]))
        shutil.copy(os.path.join(src_path, file[35:]), os.path.join(dst_path, file[35:-13]))