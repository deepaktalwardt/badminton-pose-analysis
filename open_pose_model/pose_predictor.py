import argparse
import time
import os
import cv2
print(os.getcwd())

from open_pose_model.processing import extract_parts,draw
from open_pose_model.config_reader import config_reader
from open_pose_model.model.cmu_model import get_testing_model
import json
from os import listdir,makedirs
from os.path import isfile, join
import pandas as pd


class Pose_predictor:
    def __init__(self,model_wights_file):
        self.model_wights_file = model_wights_file
        self.model = get_testing_model()
        self.model.load_weights(self.model_wights_file)
        self.params,self.model_params = config_reader()
        self.df = None
        self.input_image = None


    def single_predictor(self,img_file,output_path,csv_file):
        tic = time.time()
        print('start processing...')
        result = {}
        self.input_image = cv2.imread(img_file)
        file_name = img_file.split('/')[-1].split('.')[0]
        self.df = pd.read_csv(csv_file,header=None)
     
        [y_min,x_min,y_max,x_max] = self.df[self.df[0].str.contains(file_name+'.png')][self.df.columns[2:]].squeeze().tolist()
        crop_img = self.input_image[y_min:y_max,x_min:x_max]
        all_peaks, subset, candidate = extract_parts(crop_img, self.params, self.model, self.model_params)
        #all_peaks

        toc = time.time()
        print('processing time is %.5f' % (toc - tic))
        result[file_name] = []
        anvas = draw(crop_img, all_peaks, subset, candidate)

        for i,pos in zip(range(18),self.model_params['part_str']):
            pos = pos.replace('[', '')
            pos = pos.replace(']','')
            
            for j in range(len(all_peaks[i])):      
                #print("old all_peaks",all_peaks[i][j])          
                all_peaks[i][j] = (all_peaks[i][j][0]+x_min,all_peaks[i][j][1]+y_min)
                #all_peaks[i][j][1] =   
                #print("new all_peaks",all_peaks[i][j])
                a = int(all_peaks[i][j][0])
                b = int(all_peaks[i][j][1])
                if len(result[file_name]) >= j+1:
                    result[file_name][j][pos] = (a,b)
                else:
                    result[file_name].append({pos:(a,b)})
        for i in range(len(candidate)):
            #print(i)

            candidate[i, 0] = candidate[i, 0] + x_min
            candidate[i, 1] = candidate[i, 1] + y_min
        #print("final all_peaks", all_peaks)
        canvas = draw(self.input_image, all_peaks, subset, candidate)
        cv2.imwrite(output_path+'/'+file_name+'.png', canvas)
        cv2.destroyAllWindows()
        return result,anvas

    def batch_predictor(self,img_path,output_path,csv_file,json_file):
        result = {}
        #file_list = listdir(img_path)
        for filename in listdir(img_path):
            if not filename.endswith(".png"):
                continue
            tic = time.time()
            img_file = join(img_path, filename)
            print('start processing...')
            self.input_image  = cv2.imread(img_file)
            self.df = pd.read_csv(csv_file,header=None)
            file_name = img_file.split('/')[-1].split('.')[0]
            print(file_name)
            [y_min,x_min,y_max,x_max] = self.df[self.df[0].str.contains(file_name+'.png')][self.df.columns[2:]].squeeze().tolist()
            crop_img = self.input_image[y_min:y_max,x_min:x_max]
            all_peaks, subset, candidate = extract_parts(crop_img, self.params, self.model, self.model_params)            
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))
            result[file_name] = []
            for i,pos in zip(range(18),self.model_params['part_str']):
                pos = pos.replace('[', '')
                pos = pos.replace(']','')
                for j in range(len(all_peaks[i])):
                    all_peaks[i][j] = (all_peaks[i][j][0]+x_min,all_peaks[i][j][1]+y_min)

                    a = int(all_peaks[i][j][0])
                    b = int(all_peaks[i][j][1])
                    #print(type(a))
                    if len(result[file_name]) >= j+1:
                        result[file_name][j][pos] = (a,b)
                    else:
                        result[file_name].append({pos:(a,b)})
            for i in range(len(candidate)):
                candidate[i, 0] = candidate[i, 0] + x_min
                candidate[i, 1] = candidate[i, 1] + y_min
            canvas = draw(self.input_image, all_peaks, subset, candidate)
            op_path = output_path + '/' + file_name + '_result.png'
            cv2.imwrite(op_path, canvas)

        with open(json_file, 'w') as outfile:
            json.dump(result,outfile)
        
        cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--csv', type=str, default='output_detection_info_tty.csv', help='Enter the path to csv file containing bounding box')
    parser.add_argument('--json', type=str, default='pose.csv', help='Enter the json file name for storing skeleton')

    args = parser.parse_args()
    image_path = args.image
    output = args.output
    keras_weights_file = args.model
    csv_file = args.csv
    json_file = args.json
    makedirs(output)
    predictor = Pose_predictor(keras_weights_file)

    #result = predictor.single_predictor(image_path,output,csv_file)
    #print(result)
    predictor.batch_predictor(image_path, output,csv_file,json_file)


    # tic = time.time()
    # print('start processing...')

    # # load model

    # # authors of original model don't use
    # # vgg normalization (subtracting mean) on input images
    # model = get_testing_model()
    # model.load_weights(keras_weights_file)

    # # load config
    # params, model_params = config_reader()
    
    # input_image = cv2.imread(image_path)  # B,G,R order
    
    # all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
    # canvas = draw(input_image, all_peaks, subset, candidate)

    # result = {}
    # result['filename'] = []
    # for i,pos in zip(range(18),model_params['part_str']):
    #     pos = pos.replace('[', '')
    #     pos = pos.replace(']','')
    #     for j in range(len(all_peaks[i])):
    #         a = int(all_peaks[i][j][0])
    #         b = int(all_peaks[i][j][1])
    #         print(type(a))

    #         if len(result['filename']) >= j+1:
    #             result['filename'][j][pos] = (a,b)
    #         else:
    #             result['filename'].append({pos:(a,b)})

    # print(result)

    # toc = time.time()
    # print('processing time is %.5f' % (toc - tic))
    # with open('data2.txt', 'w') as outfile:
    #         json.dump(result,outfile)
    # cv2.imwrite(output, canvas)

    # cv2.destroyAllWindows()



