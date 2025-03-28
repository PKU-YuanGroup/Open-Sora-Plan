import os
import json
from torch.utils.data import Dataset


class GenAIBench_Image(Dataset):
    # GenAIBench with 527 prompts
    def __init__(
            self,
            root_dir,
            meta_dir,
            ):
        self.meta_dir = meta_dir
        self.root_dir = root_dir
        self.models = 'custom'
        self.dataset = json.load(open(os.path.join(self.meta_dir, f"genai_image.json"), 'r'))
        print(f"Loaded dataset: genai_image.json")
        
        self.images = [] # list of images
        self.prompt_to_images = {}
        for prompt_idx in self.dataset:
            self.images.append({
                'prompt_idx': prompt_idx,
                'prompt': self.dataset[prompt_idx]['prompt'],
                'model': self.models,
                'image': os.path.join(self.root_dir, f"{int(prompt_idx):09d}.jpg"),
            })
            if prompt_idx not in self.prompt_to_images:
                self.prompt_to_images[prompt_idx] = []
            self.prompt_to_images[prompt_idx].append(len(self.images) - 1)
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        image_paths = [item['image']]
        image = image_paths
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    

