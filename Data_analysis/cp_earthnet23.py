import shutil
import os
from tqdm import tqdm
import json
from pathlib import Path
file = open("Data_analysis/missing_value_train_from2016.json")
data = json.load(file)

# os.mkdir("/scratch/crobin/earthnet2023_preprocessed/")
os.mkdir("/scratch/crobin/earthnet2023_preprocessed/train/")

src_path = Path("/Net/Groups/BGI/work_1/scratch/s3/earthnet/earthnet2023/train/")
dst_path = Path("/scratch/crobin/earthnet2023_preprocessed/train/")
for i, file in enumerate(tqdm(list(data.keys()))):
    if data[file]['total_missing'] / (len(data[file]["serie_missing"]) * 128 * 128) < 0.15:
        path = os.path.join(dst_path, file[35:-13])
        if not os.path.exists(path):
                os.mkdir(path)
        shutil.copy2(os.path.join(src_path, file[35:]), path)