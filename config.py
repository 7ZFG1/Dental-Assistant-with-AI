from argparse import ArgumentParser

##test images path
image_folder = "images/"

##inference config and trained model paths
config_path_dict = {
                "MaskRCNN_R101_1000ep": "./model/MASKRCNN/MASKRCNN_config.py",
                }


checkpoint_path_dict = {
                    "MaskRCNN_R101_1000ep": "./model/MASKRCNN/epoch_1000.pth",
                    }

model_name = "MaskRCNN_R101_1000ep"
config_path = config_path_dict[model_name]
checkpoint_path = checkpoint_path_dict[model_name]

##score threshold
bbox_score_threshold = 0.5

##device cuda:x or cpu
device = "cuda:0"

##classes list
classes = []
for i in range(1,9,1):
    classes.append(str(i))

##color list
colors = {"1": (255,0,0), "2": (0,255,0), "3": (0,0,255), "4": (255,255,0),
                "5": (255,0,255), "6": (255,125,50), "7": (125,255,50), "8": (50,125,255), "9": (0,0,0)}

##supported image formats
image_format_list = ["png", "jpg", "JPG", "jpeg", "bmp"]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default=image_folder ,help='Image file')
    parser.add_argument('--config_path', default=config_path, help='Config file')
    parser.add_argument('--checkpoint_path', default=checkpoint_path, help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default=device, help='Device used for inference')
    parser.add_argument(
        '--bbox_score_threshold', type=float, default=bbox_score_threshold, help='bbox score threshold')
    parser.add_argument('--classes', default=classes, help='list of labels')
    parser.add_argument('--colors', default=colors, help='list of colors')
    parser.add_argument('--image_formats', default=image_format_list, help='supported image formats')
    args = parser.parse_args()
    return args