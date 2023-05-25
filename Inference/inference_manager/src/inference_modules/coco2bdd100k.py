import yaml
from yaml.loader import SafeLoader
from pathlib import Path

class coco2bdd100k:
    def __init__(self, category:str) -> None:
        self.category = category
        mod_path = Path(__file__).parent
        with open(mod_path / 'coco2bdd100k.yaml') as f:
            load_data = yaml.load(f, Loader=SafeLoader)
            self.data = load_data[self.category]

    def convert(self, coco_idx:int) -> int:
        for num in self.data:
            if num == coco_idx+1:
                # First object detection index is 1 -> offset by 1 to match python lists
                if self.category == "object detection":
                    return self.data[num][0]-1
                else:
                    return self.data[num][0]
                
                break
        return -1