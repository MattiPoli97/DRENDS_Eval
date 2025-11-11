from pathlib import Path
import cv2

class RoboLab3D():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = None
        self.get_data_path()
        pass

    def get_data_path(self):
        self.left_images_path = Path(self.data_dir, "rect_left")
        self.right_images_path = Path(self.data_dir, "rect_right")
        self.gt_path_left = Path(self.data_dir, "GT_left", "depth_maps")
        self.gt_path_right = Path(self.data_dir, "GT_right", "depth_maps")
        self.gt_path_mask_left = Path(self.data_dir, "GT_left", "masks")
        self.gt_path_mask_right = Path(self.data_dir, "GT_right", "masks")

    def get_images(self):
        self.images_left = sorted([f for f in self.left_images_path.iterdir() if f.suffix == ".png"])
        self.images_right = sorted([f for f in self.right_images_path.iterdir() if f.suffix == ".png"])
        self.gt_images_left = sorted([f for f in self.gt_path_left.iterdir() if f.suffix == ".tiff"])
        self.gt_images_right = sorted([f for f in self.gt_path_right.iterdir() if f.suffix == ".tiff"]) 
        self.gt_masks_left = sorted([f for f in self.gt_path_mask_left.iterdir() if f.suffix == ".png"])
        self.gt_masks_right = sorted([f for f in self.gt_path_mask_right.iterdir() if f.suffix == ".png"])
   
        self.img_size = cv2.imread(self.images_left[0]).shape[:2]
        
        data_list = []

        for left_img, right_img, gt_left_img, gt_right_img, mask_left, mask_right in zip(self.images_left, self.images_right, self.gt_images_left, self.gt_images_right, self.gt_masks_left, self.gt_masks_right):
            data_list.append({
            "left_image": left_img,
            "right_image": right_img,
            "gt_left_image": gt_left_img,
            "gt_right_image": gt_right_img,
            "mask_left": mask_left,
            "mask_right": mask_right
            })
        
        return data_list
    
    def __get_size__(self):
        return self.img_size

