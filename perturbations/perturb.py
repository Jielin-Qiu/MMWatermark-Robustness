import glob
import os
import cv2
import numpy as np
import cv2
import argparse
from tqdm import tqdm

from image_perturb import *
from text_perturb import *


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--modality', type = str, default = None, help='Please choose the modality to watermark')
    parser.add_argument('--device', type = str, default = 'cuda:1', help = 'Please choose the type of device' )
    parser.add_argument('--wm_method', type = str, default = None, help = 'Please choose the type of watermark method. For image, here are the options = [dwtDctSvd, rivaGan]. For text, here are the options = [lm, robust]' )
    parser.add_argument('--model', type = str, default = None, help = 'Please choose the model of watermark method. For text, here are the options = [dwtDctSvd, rivaGan]' )
    return parser.parse_args()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def main(args):

    path_to_modality = './watermarked_data/'
    
    if args.modality == 'text':

        watermarked_text = np.load(os.path.join(path_to_modality, f'text/{args.model}_{args.wm_method}_watermarked_text.npy'), allow_pickle = True).item()
        perturbed_text = {}

        count = 0
        for key in tqdm(watermarked_text, desc = 'perturbing text...'):
            try:
                perturbed_text[key] = {}
                ### CHAR LEVEL
                perturbed_text[key]['keyboard'] = {}
                perturbed_text[key]['ocr'] = {}
                perturbed_text[key]['ci'] = {}
                perturbed_text[key]['cr'] = {}
                perturbed_text[key]['cs'] = {}
                perturbed_text[key]['cd'] = {}
                ### WORD LEVEL
                perturbed_text[key]['sr'] = {}
                perturbed_text[key]['wi'] = {}
                perturbed_text[key]['ws'] = {}
                perturbed_text[key]['wd'] = {}
                perturbed_text[key]['ip'] = {}
                ### SENTENCE LEVEL
                perturbed_text[key]['formal'] = {}
                perturbed_text[key]['causal'] = {}
                perturbed_text[key]['passive'] = {}
                perturbed_text[key]['active'] = {}
                perturbed_text[key]['back_trans'] = {}
                
                if args.wm_method == 'unigram':
                    text = watermarked_text[key]['watermarked_text_seg'][0]
                    if text == '':
                        text = watermarked_text[key]['watermarked_text_all'][0]
                else:
                    text = watermarked_text[key]

                ## CHARACTER LEVEL ## 
                rate_chunk = [3,4,5,6,7]
                for rate in rate_chunk:
                    perturbed = perturb_KeyboardAug_json(text, rate)
                    perturbed_text[key]['keyboard'][rate] = perturbed

                    perturbed = perturb_OcrAug_json(text, rate)
                    perturbed_text[key]['ocr'][rate] = perturbed

                action_chunk = ['insert','substitute','swap','delete']    
                for action in action_chunk:
                    for rate in rate_chunk:
                        method = str('RandomCharAug_')+ action
                        perturbed = perturb_RandomCharAug_json(text, action, rate)
                        if action == 'insert':
                            perturbed_text[key]['ci'][rate] = perturbed
                        elif action == 'substitute':
                            perturbed_text[key]['cr'][rate] = perturbed
                        elif action == 'swap':
                            perturbed_text[key]['cs'][rate] = perturbed
                        elif action == 'delete':
                            perturbed_text[key]['cd'][rate] = perturbed
                

                # ## WORD LEVEL ## 
                for rate in rate_chunk:
                    curr_rate = 0.05 * rate
                    perturbed = insert_punc(text, curr_rate)
                    perturbed_text[key]['ip'][rate] = perturbed
                
                for rate in rate_chunk:
                    curr_rate = 0.05*rate
                    
                    perturbed = eda_perturb(text, alpha_sr = curr_rate, alpha_ri=0.0, alpha_rs=0.0, p_rd=0.0, num_aug=1)
                    perturbed_text[key]['sr'][rate] = perturbed

                    perturbed = eda_perturb(text, alpha_sr = 0.0, alpha_ri=curr_rate, alpha_rs=0.0, p_rd=0.0, num_aug=1)
                    perturbed_text[key]['wi'][rate] = perturbed

                    perturbed = eda_perturb(text, alpha_sr = 0.0, alpha_ri=0.0, alpha_rs=curr_rate, p_rd=0.0, num_aug=1)
                    perturbed_text[key]['ws'][rate] = perturbed

                    perturbed = eda_perturb(text, alpha_sr=0.0, alpha_ri=0.0, alpha_rs=0.0, p_rd=curr_rate, num_aug=1)
                    perturbed_text[key]['wd'][rate] = perturbed


                ## SENTENCE LEVEL ## 
                style_chunk = [0, 1, 2, 3]
                for style_value in style_chunk:
                    perturbed = text_style_perturb(text, style_value)
                    for rate in rate_chunk:
                        if style_value == 0:
                            perturbed_text[key]['formal'][rate] = perturbed
                        elif style_value == 1:
                            perturbed_text[key]['causal'][rate] = perturbed
                        elif style_value == 2:
                            perturbed_text[key]['passive'][rate] = perturbed
                        elif style_value ==3:
                            perturbed_text[key]['active'][rate] = perturbed
                
                perturbed = perturb_back_trans_json(text)
                for rate in rate_chunk:
                    perturbed_text[key]['back_trans'][rate] = perturbed

                count +=1
                if count == 1000:
                    break

            except:
                print('could not do it')

        np.save(f'./perturbed_data/text/{args.model}_{args.wm_method}_perturbed.npy', perturbed_text)


    elif args.modality == 'image':
        method_chunk = [gaussian_noise,shot_noise,impulse_noise,speckle_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,
                snow, frost, fog,brightness, contrast,elastic_transform, pixelate,jpeg_compression]
        
        severity_chunk = [1,2,3,4,5]
        watermarked_images = glob.glob(os.path.join(path_to_modality, f'{args.model}/{args.wm_method}/*'))
        
        for path in tqdm(watermarked_images, desc = 'Perturbing Watermarked Images: '):
            file_name = path.split('/')[-1]
            img = cv2.imread(path)
            print(img.shape)

            count_method = 0
            
            for method in method_chunk:
                for tmp in severity_chunk:

                    # input()
                    new_img= method(img, severity = tmp)
                    ensure_directory_exists(f'perturbed_data/{args.model}/{args.wm_method}/{count_method}/{tmp}')
                    cv2.imwrite(f'./perturbed_data/{args.model}/{args.wm_method}/{count_method}/{tmp}/{file_name}', new_img)
                    
                
                count_method +=1

if __name__ == '__main__':
    args=  get_args()

    main(args)



