import cv2
import os
import paddle
import paddle.fluid as fluid
import numpy as np
import argparse
from basic_model import BasicModel
from basic_data_preprocessing import InferAugmentation


parser = argparse.ArgumentParser()
parser.add_argument('--imagelist',type=str, default="")
parser.add_argument('--modelpath', type=str, default=0)
parser.add_argument('--image_folder', type=str, required=True)

args = parser.parse_args()

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)


def inference_resize()



def inference_sliding()



def inference_multi_scale()
    for scale in scales:
        image = ReScale(image)
        logits = Model(image)
        if flip:
            image_flip = Flip(image)
            logits_flip = Model(image_flip)
        
        pred.append(ScaleBack(logits))
        if flip:
            pred.append(ScaleBack(Flip(logits_flip)))

    pred = mean(pred)
    score = softmax(pred)
    pred = argmax(pred)


def save_images



# this inference code reads a list of image path, and do prediction for each image one by one
def main(args):
    # 0. env preparation
    with fluid.dygraph.guard(fluid.CPUPlace()):
        # 1. create model
        model = BasicModel()
        model.eval()
        # 2. load pretrained model 
        params_dict, _ = fluid.load_dygraph(args.model_path)
        model.load_dict(params_dict)
        # 3. read test image list
        data_list = []
        with open(args.imagelist, 'r') as infile:
            for line in infile:
                data_path = os.path.join(args.image_folder, line.split()[0])
                label_path = os.path.join(args.image_folder, line.split()[1])
                data_list.append((data_path, label_path))

        # 4. create transforms for test image, transform should be same as training
        transforms = InferAugmentation()

        color_list = []
        with open('pascal_context_colors.txt', 'r') as colorfile:
            for line in infile:
                ll = line.split()
                color_list.append(int(ll[0]))
                color_list.append(int(ll[1]))
                color_list.append(int(ll[2]))

        # 5. loop over list of images
        for index, data in enumerate(data_list):
            # 6. read image and do preprocessing
            
            image = cv2.imread(data[0], cv2.IMREAD_COLOR)
            image = cv2.cvtColor(data, cv2.COLOR)BGR2RGB
            image = image[np.newaxis,:,:,:]
            # 7. image to variable
            image = fluid.dygraph.to_variable(image)
            image = fluid.layers.transpose(image, [0, 3, 1, 2])
            # 8. call inference func
            image = model(image)
            # 9. save results
            image = fluid.layers.squeeze(image)

            pred = colorize(image.numpy() ,color_list)

            cv2.imwrite(str(index)+ '_.png', pred)

            save_blend_image(data[0], 'tmp.png')

if __name__ == "__main__":
    main(args)
