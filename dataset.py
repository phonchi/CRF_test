import os

from torch.utils.data import Dataset
from PIL import Image
import random


class VOCSegmentation(Dataset):

    def __init__(self, voc_root, year="2012", transforms=None, held_out=False, random_state=42, txt_name: str = "trainaug.txt",
                train_with_held_out=False):
        super(VOCSegmentation, self).__init__()

        root = os.path.join(voc_root, f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClassAug')

        txt_path = os.path.join(root, f'ImageSets/Segmentation/{txt_name}')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        
        if held_out is not None:
            assert isinstance(held_out, int)
            num_images = len(self.images)
            random.seed(random_state)
            smpl = random.sample(list(enumerate(self.images)), held_out)
            indices = [idx for idx, value in smpl]
            self.held_out_images = [self.images[idx] for idx in indices]
            self.held_out_masks = [self.masks[idx] for idx in indices]
            self.imags = []
            
            for idx in sorted(indices, reverse=True):
                del self.images[idx]
                del self.masks[idx]
                
            if train_with_held_out:
                self.images = self.held_out_images
                self.masks = self.held_out_masks
                
            assert (len(self.images) == len(self.masks))
            assert (len(self.held_out_images) == len(self.held_out_masks))
            
        self.transforms = transforms

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):

        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

class VOCSegmentationVal(Dataset):

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "val.txt"):
        super(VOCSegmentationVal, self).__init__()

        root = os.path.join(voc_root, f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, f'ImageSets/Segmentation/{txt_name}')
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs