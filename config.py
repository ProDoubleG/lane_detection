# directory
TUSIMPLE_DIR   = "/home/tusimple_data/TUSimple"
MODEL_SAVE_DIR = "/home/tusimple_data/model_analysis/"
SET_CLF_DS_DIR = "/home/tusimple_data/classifier"

# overall setting
SEED = 3267
NUM_ARM = 6
MODEL_NAME = f"multi_arm_{NUM_ARM}"

# dataset
INIT_PADDING = 80
RESIZE_HEIGHT = 720
RESIZE_WIDTH  = 1280

HEIGHT_TOP_CROP = 160
HEIGHT_BOTTOM_CROP = 720
NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP, 10))
ENLENGTHEN_NUM_LANE_POINT = len(range(HEIGHT_TOP_CROP, HEIGHT_BOTTOM_CROP-5, 5))

# point regressor & classifier
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
PATIENCE_LIMIT = 3
PT_CLF_THRES = 0.3

# patch generation
PATCH_WIDTH = 50
TILE_HORIZONTAL_NUM = 7
TILE_VERTICAL_NUM   = 8
CLF_PATCH_RADIUS = 25

# set classifier
SET_CLF_THRES = 0.5

