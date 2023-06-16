"""
AI Result:

type: Dict

Example:

[["label": "1", "bbox": [x1,y1,x2,y2], "poly": [x1,y1,x2,y2...xn,yn], "conf": 90]
[]
...
[]]

"""
import numpy as np
import time

from config import parse_args
from mmdet.apis import inference_detector, init_detector

from dental_postprocess import PostProcess

class AIEngine:
    def __init__(self):
        self.args = parse_args()
        self.postprocess = PostProcess()

        self.all_points_dict=[]

        self.classes = self.args.classes
        self.all_points = []

        self.bbox_score_threshold = self.args.bbox_score_threshold

        self.model = init_detector(self.args.config_path, self.args.checkpoint_path, device=self.args.device)

    def __call__(self, image):
        self.image4draw = image.copy()
        self.h,self.w,_ = image.shape

        self.start_time = time.time()
        result = inference_detector(self.model, image)

        self._get_points_only_bbox(result)
        self._calc_FPS()
        return self.all_points_dict
                
    def _calc_FPS(self):
        finish_time = time.time()
        fps = 1/(finish_time-self.start_time)
        print("FPS: ", fps)

    def _get_points(self, result):
        self.all_points = []

        bbox_result, segm_result = result

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)\
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        det_labels_impt = np.where(bboxes[:, -1] > self.bbox_score_threshold)[0]

        #FIXME: segmentation points couldnt implemented
        seg_points = np.vstack(segm_result)
        seg_points = seg_points.transpose(1,2,0)
        ##############

        classes = self.classes
        labels_impt_list = [labels[i] for i in det_labels_impt]
        labels_class = [classes[i] for i in labels_impt_list]

        for idx in range(len(bboxes[det_labels_impt])):
            x1,y1,x2,y2,conf = bboxes[idx]

            label = labels_class[idx]

            #FIXME: segmentation points couldnt implemented
            seg_img = seg_points[:,:,idx]
            seg_img = np.uint8(seg_img)*255

            seg_poly_points = np.where(seg_img==255)
            seg_poly_points = np.column_stack([seg_poly_points[1],seg_poly_points[0]])
            ##############

            tmp_points = [label,[x1,y1,x2,y2],seg_poly_points,conf] 
            self.all_points.append(tmp_points)


    def _get_points_only_bbox(self, result):
        self.all_points = []
        self.all_points_dict = []

        bbox_result, _ = result

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)\
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        det_labels_impt = np.where(bboxes[:, -1] > self.bbox_score_threshold)[0]

        classes = self.classes
        labels_impt_list = [labels[i] for i in det_labels_impt]
        labels_class = [classes[i] for i in labels_impt_list]

        for idx in range(len(bboxes[det_labels_impt])):
            x1,y1,x2,y2,conf = bboxes[idx]

            ##eleminate false detection
            if not self.postprocess.eleminate_bbox_by_shape(self.image4draw.shape, bboxes[idx]):
                continue

            label = labels_class[idx]

            seg_poly_points = []

            tmp_points = [label,[x1,y1,x2,y2],seg_poly_points,conf] 
            self.all_points.append(tmp_points)

            self.points_dict = dict(
                label=label,
                bbox=[x1,y1,x2,y2],
                poly=seg_poly_points,
                conf=conf
            )
            self.all_points_dict.append(self.points_dict)

