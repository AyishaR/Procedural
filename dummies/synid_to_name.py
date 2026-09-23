import json

imagenet100txt = "../imagenet100.txt"
imagenet_names = "imagenet_names.json"

with open(imagenet100txt, "r") as f:
    lines = f.readlines()
    synids = []
    for line in lines:
        synid = line.strip().split()
        synids.append(synid[0])

with open(imagenet_names, "r") as f:
    synid_to_name = json.load(f)

synid_to_name = {synid: (i, name) for i, (synid, name) in synid_to_name.items() if synid in synids}

assert len(synid_to_name) == len(synids), "Not all synids have names"

for synid, (i, name) in synid_to_name.items():
    print(f"{synid} ({i}): {name}")