class PostProcess:
    def __init__(self):
        self.teeth_template = ["1","2","3","4","5","6","7","8"]
        self.missing_teeth_dict = {}

    def make_area(self, all_points_dict, shape):
        self.h, self.w, _ = shape

        all_points_dict_with_areas=[]
        for data_info in all_points_dict:
            label = data_info["label"]

            x1,y1,x2,y2 = data_info["bbox"]
            middle_x, middle_y = int((x2-x1)/2)+x1, int((y2-y1)/2)+y1
            area_number = self._area_decider(middle_x,middle_y)
    
            new_data_info = dict(
                label=label,
                bbox=[x1,y1,x2,y2],
                poly=data_info["poly"],
                conf=data_info["conf"],
                area=str(area_number)
            )
            all_points_dict_with_areas.append(new_data_info)
        return all_points_dict_with_areas

    def _area_decider(self, mid_x, mid_y):
        dist_list=[]
        ##area 1
        dist_list.append(self._calc_dist(0,0,mid_x,mid_y))
        ##area 2
        dist_list.append(self._calc_dist(self.w,0,mid_x,mid_y))
        ##area 3
        dist_list.append(self._calc_dist(self.w,self.h,mid_x,mid_y))
        ##area 4
        dist_list.append(self._calc_dist(0,self.h,mid_x,mid_y))

        shortest_dist = sorted(dist_list)[0]
        area = dist_list.index(shortest_dist) + 1
        return area

    def _calc_dist(self,x1,y1,x2,y2):
        return ((x2-x1)**2 + (y2-y1)**2)**0.5
    
    def detect_missing_teeth(self, all_points_dict_with_areas):
        area_1=[]
        area_2=[]
        area_3=[]
        area_4=[]
        for data_info in all_points_dict_with_areas:
            if int(data_info["label"])>8:
                continue

            if data_info["area"] == "1":
                title = 9-int(data_info["label"])
                area_1.append(title)
            elif data_info["area"] == "2":
                title = 9-int(data_info["label"])
                area_2.append(title)
            elif data_info["area"] == "3":
                title = 9-int(data_info["label"])
                area_3.append(title)
            elif data_info["area"] == "4":
                title = 9-int(data_info["label"])
                area_4.append(title)
            else:
                raise Exception("ERROR")

        self.missing_teeth_dict["1"] = self._find_missing_teeth(area_1)
        self.missing_teeth_dict["2"] = self._find_missing_teeth(area_2)
        self.missing_teeth_dict["3"] = self._find_missing_teeth(area_3)
        self.missing_teeth_dict["4"] = self._find_missing_teeth(area_4)
        return self.missing_teeth_dict

    def _find_missing_teeth(self, lst):
        arr=[]
        if not lst==[]:
            for elem in self.teeth_template:
                if not int(elem) in lst:
                    arr.append(elem)
        else:
            arr = self.teeth_template
        return arr
    
    def eleminate_bbox_by_shape(self, shape, bbox):
        x1,y1,x2,y2,_ = bbox
        h,w,d = shape

        bbox_area = (x2-x1)*(y2-y1)
        image_area = h*w

        ratio = bbox_area/image_area
        # print("---RATIO: ", ratio)
        if ratio>0.0032:
            return True
        else:
            return False
    
    def count_teeth(self, all_points_dict_with_areas):
        return len(all_points_dict_with_areas)
