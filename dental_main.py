import cv2
import numpy as np
import os

from AI_Engine import AIEngine
from dental_postprocess import PostProcess

from config import parse_args

class DentalDeploy:
    def __init__(self):
        self.args = parse_args()
        self.image_folder = self.args.img
        self.image_dir = os.listdir(self.image_folder)

        self.image_formats = self.args.image_formats

        self.color = self.args.colors
        
        self.model = AIEngine()
        self.post_process = PostProcess()

    def __call__(self):
        for img_name in self.image_dir:
            if img_name.split(".")[-1] in self.image_formats:
                self.image_name = img_name
                image = cv2.imread(self.image_folder + img_name)
                self.image4draw = image.copy()
                self.h,self.w,_ = image.shape

                ##inference
                ai_result = self.model(image)

                ##postproccess
                all_points_dict_with_areas = self.post_process.make_area(ai_result, image.shape)
                missing_teeth_dict = self.post_process.detect_missing_teeth(all_points_dict_with_areas)

                ##postproccess and make result txt
                self._visuliaze(image, all_points_dict_with_areas, missing_teeth_dict)

    def _visuliaze(self, image, all_points_dict_with_areas, missing_teeth_dict):
        for data_info in all_points_dict_with_areas:
            x1,y1,x2,y2 = data_info["bbox"]

            try:
                color = self.color[data_info["label"]]
            except:
                #Baby teeth
                color = self.color["9"]

            ##write labels
            new_label = str(9 - int(data_info["label"]))
            if int(new_label)>8:
                new_label=data_info["label"] 
            cv2.putText(image,data_info["area"]+new_label, (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            ##draw bboxes
            self.image4draw = cv2.rectangle(self.image4draw,(int(x1),int(y1)),(int(x2),int(y2)),color, 4)

            ##draw poly
            poly_points = data_info["poly"]
            poly_points = np.array([poly_points], np.int32)
            tmp_poly_points = poly_points.reshape((-1,1,2))
            self.image4draw = cv2.polylines(self.image4draw, [tmp_poly_points], True, color, 1)

        alpha=0.4
        image = cv2.addWeighted(self.image4draw, alpha, image, 1 - alpha, 0)

        ##write result to txt
        result_file = open("dental_report.txt", "a+")
        result_file.writelines("----IMAGE: " + self.image_name+"----\n")
        result_file.writelines("Missing Teeth: ")
        missing_teeth_cnt=0
        for area in range(1,5):
            for missing_tooth in missing_teeth_dict[str(area)]:
                result_file.writelines(str(area)+missing_tooth+" ")
                missing_teeth_cnt += len(missing_tooth)
        #result_file.writelines("\nTotal Number of Teeth: " + str(self.post_process.count_teeth(all_points_dict_with_areas))+"\n\n")
        result_file.writelines("\nTotal Number of Teeth: " + str(32-missing_teeth_cnt)+"\n\n")
        result_file.close()
        
        cv2.imshow("PRED", cv2.resize(image, (int(image.shape[1]/2),int(image.shape[0]/2))))
        cv2.waitKey(0)

if __name__ == "__main__":
    deploy = DentalDeploy()
    deploy()