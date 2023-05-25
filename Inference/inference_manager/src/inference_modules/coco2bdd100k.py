import yaml
from yaml.loader import SafeLoader
from pathlib import Path

class coco2bdd100k:
    def __init__(self, category:str) -> None:
        mod_path = Path(__file__).parent
        with open(mod_path / 'coco2bdd100k.yaml') as f:
            load_data = yaml.load(f, Loader=SafeLoader)
            self.data = load_data[category]

    def convert(self, coco_idx:int) -> int:
        for num in self.data:
            if num == coco_idx+1:
                return self.data[num][0]-1
                break
        return -1