
import time
from argparse import ArgumentParser
import earthnet_models_pytorch as emp
import numpy as np
from tqdm import tqdm
from pathlib import Path
import asyncio
from tqdm.contrib.concurrent import thread_map, process_map

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

class Batch():

    def __init__(self, data_path = "/scratch/earthnet2021x/", out_path = "/scratch/earthnet2021x/", dataset = "train"):
        self.dataset = emp.setting.en21x_data.EarthNet2021XDataset(folder = Path(data_path)/dataset, dl_cloudmask = True, allow_fastaccess = False)

        self.out_path = Path(out_path)/f"{dataset}_fastaccess"

    def save_one_batch(self,i):
        filepath = self.dataset.filepaths[i]

        tile = filepath.parent.stem
        name = filepath.stem
        fastfilepath = self.out_path/tile/f"{name}.npz"
        if fastfilepath.is_file():
            return

        batch = self.dataset[i]

        fastfilepath.parent.mkdir(parents = True, exist_ok=True)

        np.savez(fastfilepath, sen2arr = batch["dynamic"][0].numpy(), eobsarr = batch["dynamic"][1].numpy(), sen2mask = batch["dynamic_mask"][0].numpy(), staticarr = batch["static"][0].numpy(), lc = batch["landcover"].numpy())


def generate_data(data_path = "/scratch/earthnet2021x/train/", out_path = "/scratch/earthnet2021x/train_fastaccess/", dataset = "train"):

    print("Generating Fast Access Data")

    B = Batch(data_path=data_path, out_path=out_path, dataset = dataset)
    
    
    results_processes = process_map(B.save_one_batch, range(len(B.dataset)),chunksize = 20)
    print("Done!")


if __name__ == "__main__":

    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path', type = str)
    parser.add_argument('out_path', type = str)
    parser.add_argument('dataset', type = str)

    args = parser.parse_args()

    generate_data(data_path = args.data_path, out_path=args.out_path, dataset=args.dataset)