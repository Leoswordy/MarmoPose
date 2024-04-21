import json
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import cv2
import sleap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SleapToCoco(ABC):
    image_id = 1
    annotation_id = 1
    type_count = 1

    def __init__(self, file_name, json_save_path, img_save_path, img_prefix=None):
        self.labels = sleap.load_file(file_name)

        Path(json_save_path).parent.mkdir(parents=True, exist_ok=True)
        self.json_save_path = json_save_path
        
        Path(img_save_path).mkdir(parents=True, exist_ok=True)
        self.img_save_path = img_save_path
        self.img_prefix = img_prefix

    def generate_skeleton(self):
        skeleton_dict = defaultdict(list)
        skeleton = self.labels.skeleton

        # Extract keypoints
        skeleton_dict['keypoints'] = [node.name for node in skeleton.nodes]

        # Map keypoint names to IDs
        name_to_id = {name: i for i, name in enumerate(skeleton.nodes)}
        for edge in skeleton.edges:
            skeleton_dict['skeleton'].append([name_to_id[edge[0]], name_to_id[edge[1]]])

        return dict(skeleton_dict)

    def generate_instances(self):
        images = []
        annotations = []
        for lf in self.labels:
            video_path = lf.video.filename
            frame_idx = lf.frame_idx
            
            # Save image and add image info
            img_name = f'{self.img_prefix}_{self.type_count:04}.jpg'
            height, width = self.save_frame(video_path, frame_idx, f'{self.img_save_path}/{img_name}')
            images.append({
                'id': self.image_id,
                'file_name': img_name,
                'height': height,
                'width': width
            })

            # Save annotations
            for instance in lf.instances:
                annotations.append(self.generate_annotation(instance))

            SleapToCoco.image_id += 1
            SleapToCoco.type_count += 1
        
        logger.info(f'{SleapToCoco.image_id} images | {SleapToCoco.annotation_id} annotations | {SleapToCoco.type_count} counts')
        
        return images, annotations
    
    def save_frame(self, video_path, frame_index, save_path, crop_coords=None):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        _, frame = cap.read()

        if crop_coords:
            x, y, width, height = crop_coords
            frame = frame[y:y+height, x:x+width]

        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cap.release()

        logger.info(f'{frame_index}th frame in {video_path} saved to {save_path}')
        
        return frame.shape[0], frame.shape[1]
    
    def generate_annotation(self, instance):
        kpt_list = []
        num_keypoints = 0
        for pt in instance.points:
            if pt.visible:
                kpt_list.extend([pt.x, pt.y, 2])
                num_keypoints += 1
            else:
                kpt_list.extend([0, 0, 0])

        bbox = self.get_bbox(instance)
        area = self.get_area(bbox)
        
        category_id = self.track_name_to_category_id(instance)

        annotation = {
            'keypoints': kpt_list,
            'image_id': self.image_id,
            'id': self.annotation_id,
            'num_keypoints': num_keypoints,
            'category_id': category_id,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        }
        SleapToCoco.annotation_id += 1

        return annotation
    
    def get_bbox(self, instance, mode='xywh'):
        assert mode in ['xywh', 'xyxy'], 'mode should be xywh or xyxy'

        x = min([pt.x for pt in instance.points])
        y = min([pt.y for pt in instance.points])

        if mode == 'xywh':
            width = max([pt.x for pt in instance.points]) - x
            height = max([pt.y for pt in instance.points]) - y

            return [x, y, width, height]
        elif mode == 'xyxy':
            x_max = max([pt.x for pt in instance.points])
            y_max = max([pt.y for pt in instance.points])

            return [x, y, x_max, y_max]
    
    def convert(self):
        images, annotations = self.generate_instances()

        json_dict = {
            'info': self.generate_info(),
            'images': images,
            'annotations': annotations,
            'categories': self.generate_categories()
        }

        with open(self.json_save_path, 'w') as f:
            json.dump(json_dict, f, indent=4)
        
        logger.info(f'Save COCO format json file to {self.json_save_path}')
    
    @abstractmethod
    def generate_info(self):
        pass
    
    @abstractmethod
    def generate_categories(self):
        pass
    
    @abstractmethod
    def track_name_to_category_id(self, instance):
        pass

    @abstractmethod
    def get_area(self, bbox):
        pass


class PairMarmosetToCoco(SleapToCoco):
    @staticmethod
    def generate_info():
        return {
            'decsription': 'MarmoPose Generated by THBI', 
            'version': '1.0',
            'year': '2024',
            'date_created': '2024/03/08'
        }

    def generate_categories(self):
        category_1 = {
            'supercategory': 'marmoset',
            'id': 1,
            'name': 'white_head_marmoset'
        }
        category_1.update(self.generate_skeleton())

        category_2 = {
            'supercategory': 'marmoset',
            'id': 2,
            'name': 'blue_head_marmoset'
        }
        category_2.update(self.generate_skeleton())
        return [category_1, category_2]

    def track_name_to_category_id(self, instance):
        category_id = 1
        if instance.track:
            category_id = 2 if instance.track.name == '1' else 1
        
        return category_id
    
    def get_area(self, bbox):
        return bbox[2] * bbox[3] / 2.0 # Since marmoset have long tail, the segmentation area is about half of the bbox area


class FamilyMarmosetToCoco(SleapToCoco):
    @staticmethod
    def generate_info():
        return {
            'decsription': 'MarmoPose Generated by THBI', 
            'version': '1.0',
            'year': '2024',
            'date_created': '2024/03/08'
        }
    
    def generate_categories(self):
        category_1 = {
            'supercategory': 'marmoset',
            'id': 1,
            'name': 'white_head_marmoset'
        }
        category_1.update(self.generate_skeleton())

        category_2 = {
            'supercategory': 'marmoset',
            'id': 2,
            'name': 'blue_head_marmoset'
        }
        category_2.update(self.generate_skeleton())

        category_3 = {
            'supercategory': 'marmoset',
            'id': 3,
            'name': 'green_head_marmoset'
        }
        category_3.update(self.generate_skeleton())

        category_4 = {
            'supercategory': 'marmoset',
            'id': 4,
            'name': 'red_head_marmoset'
        }
        category_4.update(self.generate_skeleton())

        return [category_1, category_2, category_3, category_4]
    
    def track_name_to_category_id(self, instance):
        category_id = 1
        if instance.track:
            if instance.track.name == '1':
                category_id = 2
            elif instance.track.name == '2':
                category_id = 1
            elif instance.track.name == '3':
                category_id = 3
            elif instance.track.name == '4':
                category_id = 4
        
        return category_id
    
    def get_area(self, bbox):
        return bbox[2] * bbox[3] / 2.0
    

def combine_json(json_files, save_path):
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if i == 0:
                combined_data = data
            else:
                combined_data['images'].extend(data['images'])
                combined_data['annotations'].extend(data['annotations'])
    
    with open(save_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    logger.info(f'Save combined json file to {save_path}')


def train_test_split(json_file, test_ratio=0.2, split_seed=42):
    """
    Split the json file into 
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    info = data['info']
    images = data['images']
    annotations = data['annotations']

    images.sort(key=lambda x: x['id'])
    annotations.sort(key=lambda x: x['image_id'])

    split = int(len(images) * test_ratio)
    random.seed(split_seed)
    random_choose = random.sample(range(len(images)), split)

    train_images = []
    test_images = []
    train_annotations = []
    test_annotations = []

    for image in images:
        if image['id'] in random_choose:
            test_images.append(image)
        else:
            train_images.append(image)
    
    for ann in annotations:
        if ann['image_id'] in random_choose:
            test_annotations.append(ann)
        else:
            train_annotations.append(ann)

    train_file_name = json_file.replace('all.json', 'train.json')
    test_file_name = json_file.replace('all.json', 'test.json')

    train_data = {
        'categories': categories,
        'info': info,
        'images': train_images,
        'annotations': train_annotations
    }

    with open(train_file_name, 'w') as f:
        json.dump(train_data, f)

    logger.info(f'Train set: {len(train_images)} images | {len(train_annotations)} annotations')
    logger.info(f'Save train set to {train_file_name}')


    test_data = {
        'categories': categories,
        'info': info,
        'images': test_images,
        'annotations': test_annotations
    }

    with open(test_file_name, 'w') as f:
        json.dump(test_data, f)
    
    logger.info(f'Test set: {len(test_images)} images | {len(test_annotations)} annotations')
    logger.info(f'Save test set to {test_file_name}')


def run_convert_pair():
    file_name = r'D:\ccq\MarmoPose\data\pair.slp'
    json_save_path=r'D:\ccq\MarmoPose\data\marmoset\annotations\all.json'
    img_save_path=r'D:\ccq\MarmoPose\data\marmoset\images'

    converter = PairMarmosetToCoco(file_name, json_save_path, img_save_path, img_prefix='pair')
    converter.convert()

    # Split this json file into train and test set
    train_test_split(json_save_path, test_ratio=0.2, split_seed=42)


def run_convert_family():
    file_name = r'D:\ccq\MarmoPose\data\family.slp'
    json_save_path = r'D:\ccq\MarmoPose\data\marmoset_family\annotations\all.json'
    img_save_path = r'D:\ccq\MarmoPose\data\marmoset_family\images'

    converter = FamilyMarmosetToCoco(file_name, json_save_path, img_save_path, img_prefix='family')
    converter.convert()

    # Split this json file into train and test set
    train_test_split(json_save_path, test_ratio=0.2, split_seed=42)


if __name__ == '__main__':
    """
    This is the default convert function for videos with 2 marmosets. Converted annotations are saved in the json file, and images are cropped and saved in the image folder.
    TODO: Modify the paths in the funtion to your own paths
    TODO: Modify the infromation in `PairMarmosetToCoco` class if your setting is different from the default
        - `generate_info`: Basic information of the dataset
        - `generate_categories`: CaCategory name and id of each marmoset
        - `track_name_to_category_id`: Mapping from track name (in SLEAP) to category id
    """
    # run_convert_pair()


    """
    This is the default convert function for videos with 4 marmosets. Converted annotations are saved in the json file, and images are cropped and saved in the image folder.
    TODO: Modify the paths in the funtion to your own paths
    TODO: Modify the infromation in `FamilyMarmosetToCoco` class if your setting is different from the default
        - `generate_info`: Basic information of the dataset
        - `generate_categories`: CaCategory name and id of each marmoset
        - `track_name_to_category_id`: Mapping from track name (in SLEAP) to category id
    """
    run_convert_family()

