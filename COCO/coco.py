import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, annotation_file):
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        self.ids = [img['id'] for img in self.data['images']]
        
        self.to_tensor = transforms.ToTensor()
        self.coco_dataset_dir = './data/coco/val2017/'
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_data = next(img for img in self.data['images'] if img['id'] == img_id)
        file_name = img_data['file_name']
        
        
        img = Image.open(os.path.join(self.coco_dataset_dir, file_name)).convert('RGB')
        img = self.to_tensor(img)
        
        
        width = img_data['width']
        height = img_data['height']
        anns = [ann for ann in self.data['annotations'] if ann['image_id'] == img_id]
        captions = [item['caption'] for item in anns]

        print(captions)
        
        return {
            'file_name': file_name,
            'image_id': img_id,
            'width': width,
            'height': height,
            'caption': captions,
            'image': img
        }