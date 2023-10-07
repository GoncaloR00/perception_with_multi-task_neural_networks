import yaml
from yaml.loader import SafeLoader
from pathlib import Path

class ade20k2bdd100k:
    def __init__(self, category:str) -> None:
        self.category = category
        mod_path = Path(__file__).parent
        with open(mod_path / 'ade20k2bdd100k.yaml') as f:
            load_data = yaml.load(f, Loader=SafeLoader)
            # print(f"Category = {self.category}")
            self.data = load_data[self.category]
            # print(f"Data = {self.data}")

    def convert(self, coco_idx:int) -> int:
        # print(self.data)
        for num in self.data:
            if num == coco_idx:
                # First object detection index is 1 -> offset by 1 to match python lists
                return self.data[num][0]
                break
        return -1
    

# converter = ade20k2bdd100k('semantic segmentation')
# print(converter.convert(60))