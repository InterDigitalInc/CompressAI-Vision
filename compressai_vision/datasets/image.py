# Copyright (c) 2022-2023, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import json

from compressai_vision.utils import logger

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import DatasetCatalog 

#from compressai.registry import register_dataset



class BaseDataset(Dataset):
    def __init__(self, dataset_name):
        self.sampler = None
        self.collate_fn = None
        self.dataset_name = dataset_name
    
    def get_dataset_name(self):
        return self.dataset_name
    
class Detectron2BasedDataset(MapDataset):
    def __init__(self, dataset_name, dataset, cfg, images_folder, annotation_path):
        self.dataset_name = dataset_name

        self.annotation_path = annotation_path
        self.images_folder = images_folder

        try:
            DatasetCatalog.get(dataset_name)
        except KeyError: 
            logger.warning(__name__, f'It seems a new dataset. Let me register the new dataset \"{dataset_name}\" for you')
            register_coco_instances(dataset_name, {}, self.annotation_path, self.images_folder)
    

        self.sampler = InferenceSampler(len(dataset))
        def bypass_collator(batch):
            return batch

        self.collate_fn = bypass_collator

        dataset = DatasetFromList(dataset, copy=False)
        mapper = DatasetMapper(cfg, False)

        super().__init__(dataset, mapper)
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_annotation_path(self):
        return self.annotation_path
    
    def get_images_folder(self):
        return self.images_folder
    


#@register_dataset("ImageFolder")
class ImageFolder(BaseDataset):
    """Load an image folder database. testing image samples
    are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - img000.png
            - img001.png
            
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        use_BGR (Bool): if True the color order of the sample is BGR otherwise RGB returned 
    """

    def __init__(self, dataset_name, root, transform=None, ret_name=False, use_BGR=False):
        super().__init__(dataset_name)
        
        splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file()]

        self.use_BGR = use_BGR
        self.transform = transform
        self.ret_name = ret_name


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        if self.use_BGR is True:
            r, g, b = img.split()
            img = Image.merge('RGB', (b,g,r))

        if self.transform:
            if self.ret_name is True:
                return (self.transform(img), str(self.samples[index]))
            return self.transform(img)

        if self.ret_name is True:
            return (img, str(self.samples[index]))

        return img

    def __len__(self):
        return len(self.samples)

#@register_dataset("SFUHW_ImageFolder")
class SFUHW_ImageFolder(Detectron2BasedDataset):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - images
                - img000.png
                - img001.png
                - imgxxx.png
            - annotations
                - xxxx.json
            
    Args:
        root (string): root directory of the dataset

    """

    def __init__(self, root, cfg, 
                 dataset_name='sfu-hw-object-v1',
                 **kwargs):
        images_dir = Path(root) / "images"
        annotations_dir = Path(root) / "annotations"

        if not images_dir.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{images_dir}"')
        
        if not annotations_dir.is_dir():
            raise RuntimeError(f'Invalid annotation directory "{annotations_dir}"')

        # load samples and annotations
        samples = [f for f in sorted(images_dir.iterdir()) if f.is_file()]
        annotations = [f for f in sorted(annotations_dir.iterdir()) if f.is_file() and f.suffix[1:] == 'json']

        if len(annotations) != 1:
            raise RuntimeError(f"The number of json file for the samples annotations must be 1, but got {len(annotations)}")

        # merge annotation and sample list
        with open(annotations[0]) as f:
            data = json.load(f)
            dataset = data['images']
            del data['images']

            assert len(dataset) == len(samples), "Number of samples listed in json is different with the number of samples in the image folder"

            # update file addresses
            convert_image_id_to_list_idx = {}
            for e, addr in enumerate(samples):
                ds = dataset[e]
                if ds['file_name'] != addr.name:
                    raise RuntimeError(f"File name mismatch. Expect to get {ds['file_name']}, but we've got {addr.name}")
                ds['file_name'] = str(addr)
                ds['image_id'] = ds['id']
                convert_image_id_to_list_idx.update({f"{ds['id']}": e})
                del ds['id']
                # create an empty list for annotations
                ds.update({"annotations":[]})
            
            # merge annotations into dataset
            for annotation in data['annotations']:
                idx = convert_image_id_to_list_idx[f"{annotation['image_id']}"]
                
                del annotation['image_id']
                del annotation['id']

                dataset[idx]['annotations'].append(annotation)

        super().__init__(dataset_name, dataset, cfg, images_dir, annotations[0])


#@register_dataset("COCO_ImageFolder")
class COCO_ImageFolder(Detectron2BasedDataset):
    """Load an image folder database with Detectron2 Cfg. testing image samples
    and annotations are respectively stored in separate directories
    (Currently this class supports none of training related operation ):

    .. code-block::
        - rootdir/
            - [train_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - [validation_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - [test_folder]
                - img000.jpg
                - img001.jpg
                - imgxxx.jpg
            - annotations
                - [instances_val].json
                - [captions_val].json
                - ...
            
    Args:
        root (string): root directory of the dataset

    """

    def __init__(self, root, cfg, 
                 dataset_name='mpeg-coco', 
                 img_path='val2017', 
                 annot_path='annotations/instances_val2017.json',
                 **kwargs):
        images_dir = Path(root) / img_path
        annotations_file = Path(root) / annot_path

        if not images_dir.is_dir():
            raise RuntimeError(f'Invalid image sample directory "{images_dir}"')
        
        if not annotations_file.is_file():
            raise RuntimeError(f'Invalid annotation file "{annotations_file}"')
        
        dataset = load_coco_json(annotations_file, images_dir, dataset_name=dataset_name)

        super().__init__(dataset_name, dataset, cfg, images_dir, annotations_file)
