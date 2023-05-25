#!/usr/bin/python3

import yaml
from yaml.loader import SafeLoader
from pathlib import Path
mod_path = Path(__file__).parent

det_classes = []
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for num in data['object detection']:
        print(num)
        print("----------------------------------------------------")
        det_classes.append(data['object detection'][num])
print(det_classes)
