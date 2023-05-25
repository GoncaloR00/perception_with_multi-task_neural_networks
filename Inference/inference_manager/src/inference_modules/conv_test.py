#!/usr/bin/python3
from coco2bdd100k import coco2bdd100k

teste = coco2bdd100k("object detection")
i=30
print(f"Number: {i} | Result: {teste.convert(i)}")