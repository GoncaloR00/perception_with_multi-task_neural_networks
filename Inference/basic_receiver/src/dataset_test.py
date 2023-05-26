#!/usr/bin/python3
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import math
mod_path = Path(__file__).parent
with open(mod_path / 'bdd100k.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
objDect_cls = {}

for name in data['object detection']:
    # objDect_cls[data['object detection'][name]] = [0,int(255/(math.sqrt(name[0]))),int(255/(math.sqrt(name[0])))]
    print(name)