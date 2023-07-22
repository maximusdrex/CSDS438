import torch
import rawpy
import os

processed_path = "processed/"

def get_train_list():
    train_list = []
    with open("Fuji_train_list.txt", 'r') as f:
        train_list = f.readlines()
        train_list = [x.split(" ") for x in train_list]
        train_list = [{"train": x[0], "truth": x[1], "iso": x[2], "aperture": x[3]} for x in train_list]
        train_list = [{"train": x["train"].split('/'), "truth": x["truth"].split('/'), "iso": x["iso"], "aperture": x["aperture"]} for x in train_list]
        train_list = [{"train": "/".join(x["train"][1:]), "truth": "/".join(x["truth"][1:]), "iso": x["iso"], "aperture": x["aperture"]} for x in train_list]
        train_list = [x for x in train_list if os.path.isfile(x["train"]) and os.path.isfile(x["truth"])]
    return train_list

train_list = get_train_list()
count = 0
if(train_list):
    for pair in train_list:
        # Change the path to the processed path and change the file extension to .pt
        t_path = processed_path + pair["train"].split('/')[-1][0:-4] + ".pt"
        gt_path = processed_path + pair["truth"].split('/')[-1][0:-4] + ".pt"
        if not os.path.isfile(t_path):
            try:
                raw = rawpy.imread(pair["train"])
                rgb = raw.postprocess()
                torch.save(torch.tensor(rgb), t_path)
                count += 1
            except:
                print("error reading file {}".format(pair["train"]))
        if not os.path.isfile(gt_path):
            try:
                raw = rawpy.imread(pair["truth"])
                rgb = raw.postprocess()
                torch.save(torch.tensor(rgb), gt_path)
                count += 1
            except:
                print("error reading file {}".format(pair["truth"]))

print("preprocessed {} files".format(count))