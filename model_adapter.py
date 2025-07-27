import dtlpy as dl
import os
import sys
import ast
import cv2
import time
import torch
import numpy as np
import logging

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict, check_video_rotation
from misc.utils import find_person_id_associations

# Set up logger
logger = logging.getLogger('[HRNet Pose Estimation]')

class HRNetModelAdapter(dl.BaseModelAdapter):

    def __init__(self, model_entity: dl.Model):
        # Available weight configurations
        self.available_weights = {
            "hrnet_w48_384x288": {
                "file": "pose_hrnet_w48_384x288.pth",
                "hrnet_c": 48,
                "hrnet_j": 17,
                "image_resolution": "(384, 288)"
            },
            "hrnet_w32_256x192": {
                "file": "pose_hrnet_w32_256x192.pth", 
                "hrnet_c": 32,
                "hrnet_j": 17,
                "image_resolution": "(256, 192)"
            },
            "hrnet_w32_256x256": {
                "file": "pose_hrnet_w32_256x256.pth",
                "hrnet_c": 32, 
                "hrnet_j": 17,
                "image_resolution": "(256, 256)"
            }
        }
        
        # Get configuration from model entity
        configuration = getattr(model_entity, 'configuration', {}) or {}
        
        # Set weight configuration based on model configuration or default
        weight_config_name = configuration.get('weight_config', 'hrnet_w48_384x288')
        if weight_config_name not in self.available_weights:
            logger.warning(f"Unknown weight configuration '{weight_config_name}'. Using default 'hrnet_w48_384x288'")
            weight_config_name = 'hrnet_w48_384x288'
        
        # Apply weight configuration
        weight_config = self.available_weights[weight_config_name]
        self.hrnet_c = weight_config['hrnet_c']
        self.hrnet_j = weight_config['hrnet_j'] 
        self.hrnet_weights = f'./weights/{weight_config["file"]}'
        self.image_resolution = weight_config['image_resolution']
        
        # Other configuration parameters with defaults
        self.hrnet_joints_set = configuration.get('hrnet_joints_set', 'coco')
        self.single_person = configuration.get('single_person', False)
        self.yolo_version = configuration.get('yolo_version', 'v5')
        self.use_tiny_yolo = configuration.get('use_tiny_yolo', False)
        self.disable_tracking = configuration.get('disable_tracking', False)
        self.max_batch_size = configuration.get('max_batch_size', 16)
        self.device = configuration.get('device', None)
        self.enable_tensorrt = configuration.get('enable_tensorrt', False)
        
        logger.info(f"Initialized HRNet with {weight_config_name} configuration")
        
        super().__init__(model_entity=model_entity)

    def download_hrnet_weights(self):
        """
        Download only the specific HRNet weight file that's needed based on configuration.
        """
        try:
            import gdown
        except ImportError:
            logger.info("Installing gdown...")
            os.system("pip install gdown")
            import gdown
        
        # Create weights directory if it doesn't exist
        weights_dir = "./weights"
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
            logger.info(f"Created directory: {weights_dir}")
        
        # Change to weights directory
        original_dir = os.getcwd()
        os.chdir(weights_dir)
        
        # Dictionary of all available weight files and their Google Drive IDs
        all_weight_files = {
            "pose_hrnet_w48_384x288.pth": "1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS",
            "pose_hrnet_w32_256x192.pth": "1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38", 
            "pose_hrnet_w32_256x256.pth": "1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v"
        }
        
        # Determine which weight file is needed based on current configuration
        weight_filename = os.path.basename(self.hrnet_weights)
        
        if weight_filename not in all_weight_files:
            logger.error(f"Unknown weight file {weight_filename}")
            available_files = list(all_weight_files.keys())
            logger.info(f"Available options: {available_files}")
            os.chdir(original_dir)
            return
        
        if os.path.exists(weight_filename):
            logger.info(f"Weight file {weight_filename} already exists")
        else:
            logger.info(f"Downloading {weight_filename}...")
            try:
                file_id = all_weight_files[weight_filename]
                gdown.download(f"https://drive.google.com/uc?id={file_id}", weight_filename, quiet=False)
                logger.info(f"Successfully downloaded {weight_filename}")
            except Exception as e:
                logger.error(f"Failed to download {weight_filename}: {str(e)}")
        
        # Return to original directory
        os.chdir(original_dir)

    def load(self, local_path, **kwargs):
        self.model = None

        # Download weights if they don't exist
        if not os.path.exists(self.hrnet_weights):
            logger.info(f"Weights not found at {self.hrnet_weights}. Downloading...")
            self.download_hrnet_weights()

        if self.device is not None:
            self.device = torch.device(self.device)
        else:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        # Convert string resolution to tuple - handle both string and tuple cases
        if isinstance(self.image_resolution, str):
            self.image_resolution = ast.literal_eval(self.image_resolution)
        
        self.has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'

        if self.yolo_version == 'v3':
            if self.use_tiny_yolo:
                yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
                yolo_weights_path = "./models_/detectors/yolo/weights/yolov3-tiny.weights"
            else:
                yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
                yolo_weights_path = "./models_/detectors/yolo/weights/yolov3.weights"
            yolo_class_path = "./models_/detectors/yolo/data/coco.names"
        elif self.yolo_version == 'v5':
            # YOLOv5 comes in different sizes: n(ano), s(mall), m(edium), l(arge), x(large)
            if self.use_tiny_yolo:
                yolo_model_def = "yolov5n"  # this  is the nano version
            else:
                yolo_model_def = "yolov5m"  # this  is the medium version
            if self.enable_tensorrt:
                yolo_trt_filename = yolo_model_def + ".engine"
                if os.path.exists(yolo_trt_filename):
                    yolo_model_def = yolo_trt_filename
            yolo_class_path = ""
            yolo_weights_path = ""
        else:
            raise ValueError('Unsopported YOLO version.')

        # Comment out or modify this line if you're not using dtlpy artifacts
        # self.model_entity.artifacts.download(local_path='./weights')

        self.model = SimpleHRNet(
            self.hrnet_c,
            self.hrnet_j,
            self.hrnet_weights,
            model_name='HRNet',
            resolution=self.image_resolution,
            multiperson=not self.single_person,
            return_bounding_boxes=not self.disable_tracking,
            max_batch_size=self.max_batch_size,
            yolo_version=self.yolo_version,
            yolo_model_def=yolo_model_def,
            yolo_class_path=yolo_class_path,
            yolo_weights_path=yolo_weights_path,
            device=self.device,
            enable_tensorrt=self.enable_tensorrt
        )

    def predict_image(self, item: dl.Item, pose_label='person'):

        filename = item.download(local_path=f'./tmp/{item.filename}')

        image = cv2.imread(filename)

        pts = self.model.predict(image)
        boxes, pts = pts

        recipe = item.dataset.recipes.list()[0]        
        labels = joints_dict()[self.hrnet_joints_set]['keypoints']

        template_id = recipe.get_annotation_template_id(template_name=pose_label)

        builder = item.annotations.builder()

        for pt_id, pt in enumerate(pts):
            # Define the Pose parent annotation and upload it to the item
            # Note: need to have the parent annotation (the pose) created in order to link children (points) to it...
            # parent_annotation = builder.add(annotation_definition=dl.Pose(label=pose_label, template_id=template_id))
            parent_annotation = item.annotations.upload(
                dl.Annotation.new(annotation_definition=dl.Pose(label=pose_label.capitalize(),
                                                                template_id=template_id,
                                                                # instance_id is optional
                                                                instance_id=pt_id)))[0]

            for jd, p in enumerate(pt):
                x = p[1]
                y = p[0]
                label = labels[jd]

                # Add child points
                builder.add(annotation_definition=dl.Point(x=x, y=y, label=label),
                            parent_id=parent_annotation.id)
            builder.add(annotation_definition=dl.Box(top=boxes[pt_id][1],
                                                     bottom=boxes[pt_id][3],
                                                     left=boxes[pt_id][0],
                                                     right=boxes[pt_id][2],
                                                     label=pose_label.capitalize()))
        # builder.upload()
        return builder.annotations

    def predict_video(self, item: dl.Item, pose_label='person'):

        filename = item.download(local_path=f'./tmp/{item.filename}')

        if not self.disable_tracking:
            prev_boxes = None
            prev_pts = None
            prev_person_ids = None
            next_person_id = 0
        t_start = time.time()

        if filename is not None:
            rotation_code = check_video_rotation(filename)
            video = cv2.VideoCapture(filename)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            assert video.isOpened()

        video_writer = None
        persons_annotations = dict()
        persons_pose_annotations = dict()  # Store pose parent annotations for each person
        labels = joints_dict()[self.hrnet_joints_set]['keypoints']
        
        # Get recipe and template_id for pose annotations
        recipe = item.dataset.recipes.list()[0]
        template_id = recipe.get_annotation_template_id(template_name=pose_label)
        
        frame_num = 0
        while True:
            t = time.time()

            if filename is not None:
                ret, frame = video.read()
                if not ret:
                    t_end = time.time()
                    logger.info(f"\n Total Time: {t_end - t_start}")
                    break
                if rotation_code is not None:
                    frame = cv2.rotate(frame, rotation_code)
            else:
                frame = video.read()
                if frame is None:
                    break

            pts = self.model.predict(frame)

            if not self.disable_tracking:
                boxes, pts = pts

            if not self.disable_tracking:
                if len(pts) > 0:
                    if prev_pts is None and prev_person_ids is None:
                        person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                        next_person_id = len(pts) + 1
                    else:
                        boxes, pts, person_ids = find_person_id_associations(
                            boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts,
                            prev_person_ids=prev_person_ids,
                            next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4,
                            smoothing_alpha=0.1,
                        )
                        next_person_id = max(next_person_id, np.max(person_ids) + 1)
                else:
                    person_ids = np.array((), dtype=np.int32)

                prev_boxes = boxes.copy()
                prev_pts = pts.copy()
                prev_person_ids = person_ids

            else:
                person_ids = np.arange(len(pts), dtype=np.int32)

            self.validate_person_or_introduce_new(person_ids, persons_annotations, persons_pose_annotations, item, labels, frame_num, pose_label, template_id)

            for i, (pt, pid) in enumerate(zip(pts, person_ids)):
                # person = pid
                annotations = persons_annotations[pid]
                pose_annotation = persons_pose_annotations[pid]
                self.add_frame_annotation(annotations, pt, frame_num, labels, pose_annotation)

                # frame = draw_points_and_skeleton(frame, pt, joints_dict()[self.hrnet_joints_set]['skeleton'],
                #                                  person_index=pid,
                #                                  points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                #                                  points_palette_samples=10)

            frame_num = frame_num + 1
            fps = 1. / (time.time() - t)
            
            progress = (frame_num / total_frames) * 100
            logger.info(f'\rframe {frame_num}/{total_frames} (%.1f%%) - framerate: {fps:.1f} fps, for {len(pts)} person(s) ' % progress)

        output_annotations = []
        min_frame_presence = 10
        for person in persons_annotations:
            for annotation in persons_annotations[person]:
                if len(annotation.frames)<min_frame_presence:
                    continue
                output_annotations.append(annotation)

        # os.remove(filename)
        return output_annotations

    def validate_person_or_introduce_new(self, person_ids, annotations, pose_annotations, item, labels, frame, pose_label, template_id):
        for person in person_ids:
            if not person in annotations:
                annotations[person] = []
                
                # Create and upload pose parent annotation for this person to get its ID
                pose_annotation = item.annotations.upload(
                    dl.Annotation.new(
                        annotation_definition=dl.Pose(
                            label=pose_label.capitalize(),
                            template_id=template_id,
                            instance_id=int(person)
                        ),
                        item=item,
                        object_id=int(person),
                        frame_num=frame
                    )
                )[0]
                pose_annotations[person] = pose_annotation

                # Create point annotations linked to the pose parent
                for index in range(len(labels)):
                    label = labels[index]
                    annotation = dl.Annotation.new(
                        item=item, 
                        object_id=int(person), 
                        frame_num=frame,
                        parent_id=pose_annotation.id  # Link to pose parent
                    )
                    annotations[person].append(annotation)

    def add_frame_annotation(self, video_annotations, points, frame_index, labels, pose_annotation):
        # Add frame to pose annotation
        pose_annotation.add_frame(annotation_definition=dl.Pose(
            label=pose_annotation.annotation_definition.label,
            template_id=pose_annotation.annotation_definition.template_id,
            instance_id=pose_annotation.annotation_definition.instance_id
        ), frame_num=frame_index)
        
        for i, pt in enumerate(points):
            video_annotation = video_annotations[i]
            video_annotation.add_frame(annotation_definition=dl.Point(x=pt[1], y=pt[0], label=labels[i]),
                                       frame_num=frame_index)

    def prepare_item_func(self, item: dl.Item):
        return item

    def predict(self, batch, **kwargs):
        res = []
        for item in batch:
            mimetype = item.metadata['system']['mimetype']
            if mimetype.startswith('video'):
                res.append(self.predict_video(item))
            elif mimetype.startswith('image'):
                res.append(self.predict_image(item))
        return res


if __name__ == '__main__':
    dl.setenv('rc')
    model = dl.models.get(model_id="6847091a772e642cc5683c80")
    adapter = HRNetModelAdapter(model_entity=model)
    adapter.load(local_path='./weights')
    item = dl.items.get(item_id="6846f07b4761e970c8e79d7e")
    adapter.predict_items(items=[item])