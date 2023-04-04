#!/usr/bin/python3

import yaml
from yaml.loader import SafeLoader
from pathlib import Path
mod_path = Path(__file__).parent

det_classes = []
with open(mod_path / 'coco.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    for name in data['names']:
        det_classes.append(data['names'][name])
print(det_classes)
