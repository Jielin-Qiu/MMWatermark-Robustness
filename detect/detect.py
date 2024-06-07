from imwatermark import WatermarkEncoder, WatermarkDecoder
from transformers import AutoTokenizer, InstructBlipProcessor, AutoProcessor
import argparse
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
torch.set_num_threads(2)
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from rouge import Rouge
from extended_watermark_processor import WatermarkDetector
from detect import permutation_test
from sewar.full_ref import ssim, psnr
from PIL import Image
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--modality', type = str, default = None, help='Please choose the modality to watermark')
    parser.add_argument('--device', type = str, default = 'cuda:1', help = 'Please choose the type of device' )
    parser.add_argument('--wm_method', type = str, default = None, help = 'Please choose the type of watermark method. For text, here are the options = [dwtDctSvd, rivaGan]' )
    parser.add_argument('--perturb_method', type = int, default = None, help = 'Please choose the type perturbation method')
    parser.add_argument('--temp', type = int, default = None, help = 'Please choose the temperature' )
    parser.add_argument('--model', type = str, default = None, help = 'Please choose the model' )
    return parser.parse_args()

def ensure_file_exists(file_path):
    if not os.path.exists(file_path):
        save_dic = {}
        save_dic['wm_method'] = []
        save_dic['perturb_method'] = []
        save_dic['accuracy'] =  []
        save_dic['temp'] = []
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")
        save_dic = np.load(file_path, allow_pickle = True).item()
    return save_dic

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def calculate_bit_accuracy(text1, text2):
    binary1 = text_to_binary(text1)
    binary2 = text_to_binary(text2)
    max_len = max(len(binary1), len(binary2))
    binary1 = binary1.ljust(max_len, '0')
    binary2 = binary2.ljust(max_len, '0')
    matching_bits = sum(b1 == b2 for b1, b2 in zip(binary1, binary2))

    return matching_bits / max_len

def main(args):
    path_to_modality = './perturbed_data/'
    correct = 0
    num_files = 0

    if args.modality == 'text':
        if args.model == 'blip':
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir = '../../.huggingface_cache')
            llama_tokenizer = processor.tokenizer
        if args.model == 'llava':
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir = '../../.huggingface_cache')
            llama_tokenizer = processor.tokenizer

        path_to_modality = '../../perturbed_data/'
        path_to_watermark = '../../watermarked_data/'
        watermark_detector = WatermarkDetector(vocab=list(llama_tokenizer.get_vocab().values()),
                                                gamma=0.25, # should match original setting
                                                seeding_scheme="simple_1", # should match original setting
                                                device='cpu', # must match the original rng device type
                                                tokenizer=llama_tokenizer,
                                                z_threshold=0.5,
                                                normalizers=[],
                                                ignore_repeated_ngrams=True)

        file_path = f'{path_to_modality}{args.modality}/{args.model}_{args.wm_method}_perturbed_para.npy'
        watermarked_file_path = f'{path_to_watermark}{args.modality}/{args.model}_{args.wm_method}_watermarked_text.npy'
        watermark_np_file = np.load(watermarked_file_path, allow_pickle = True).item()
        np_file = np.load(file_path, allow_pickle = True).item()
        
        save_dic = {}
        save_dic['averaged_accuracy'] = []
        save_dic['perturb_method'] = []
        save_dic['averaged_bleurt'] = []
        save_dic['averaged_rouge-l_f1'] = []
        save_dic['averaged_bit_acc'] = []

        bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
        bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

        rouge = Rouge()

        z_threshold = 6
        for img_key in tqdm(np_file.keys(), desc = 'Calculating metrics: '):
            img_file = np_file[img_key]
            
            watermarked_caption = watermark_np_file[img_key]
            
            if args.wm_method == 'unigram':
                watermarked_caption=watermarked_caption['watermarked_text_seg'][0]

            count = 0
            for perturb_method in img_file.keys():
                perturbed_img_file = img_file[perturb_method]
                correct = 0
                num_files = 0
                bleurt_scores = 0
                rouge_scores = 0
                bit_accs = 0

                for rate_chunk in perturbed_img_file.keys():
                    perturbed_text = perturbed_img_file[rate_chunk]
                    try:
                        candidates = [perturbed_text[0]]
                        references = [watermarked_caption]
                    
                        bleurt_model.eval()
                        with torch.no_grad():
                            
                            inputs = bleurt_tokenizer(references, candidates, padding='longest', return_tensors='pt')
                            
                            inputs['input_ids'] = inputs['input_ids'][:, :512]
                            inputs['attention_mask'] = inputs['attention_mask'][:, :512]
                            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
                            
                            res = bleurt_model(**inputs).logits.flatten().tolist()
                            bleurt_score = res[0]


                        score = rouge.get_scores(candidates, references)
                        score = score[0]['rouge-l']
                        f = score['f']
                        

                        bit_acc = calculate_bit_accuracy(candidates[0], references[0])
                        
                        score_dict = watermark_detector.detect(watermarked_caption)

                        if score_dict['prediction']:
                            correct +=1
                            num_files +=1
                        else:
                            num_files +=1
                        
                        bleurt_scores += bleurt_score
                        rouge_scores += f
                        bit_accs += bit_acc
                        print('num_files')
                    except:
                        num_files +=1
                        print('except')
                
                save_dic['perturb_method'].append(perturb_method)
                try:
                    save_dic['averaged_accuracy'].append(correct/num_files)
                    save_dic['averaged_bleurt'].append(bleurt_scores/num_files)
                    save_dic['averaged_rouge-l_f1'].append(rouge_scores/num_files)
                    save_dic['averaged_bit_acc'].append(bit_accs/num_files)
                except:
                    print('0')
                    save_dic['averaged_accuracy'].append(0)
                    save_dic['averaged_bleurt'].append(0)
                    save_dic['averaged_rouge-l_f1'].append(0)
                    save_dic['averaged_bit_acc'].append(0)
        
        np.save(f'{args.model}_calculated_metrics_{args.wm_method}_text_par.npy', save_dic)

    elif args.modality == 'image':
        file_path = f'./detect_wm_{args.modality}_{args.model}_2.npy'

        perturbed_images = glob.glob(os.path.join(path_to_modality, f'{args.model}/{args.wm_method}/{args.perturb_method}/{args.temp}/*'))
        
        encoder = WatermarkEncoder()
        if args.wm_method == 'rivaGan':
            encoder.loadModel()
        wm = 'test'
        
        ### detecting watermark
        decoder = WatermarkDecoder('bytes', 32)
        
        for path in tqdm(perturbed_images, desc = 'Detecting Watermarks: '):
            file_name = path.split('/')[-1]
            bgr = cv2.imread(path)
 
           # detect watermark
            bgr = cv2.imread(path)
            watermark = decoder.decode(bgr, args.wm_method)
            try:
                print(watermark.decode('utf-8'))
                if watermark.decode('utf-8') == 'test':
                    correct +=1
                    num_files +=1
                else:
                    num_files +=1
            except:
                print('Could not decode')
                num_files +=1
                pass

        print('Accuracy', correct/num_files)

        save_dic = ensure_file_exists(file_path)
        save_dic['wm_method'].append(args.wm_method)
        save_dic['temp'].append(args.temp)
        save_dic['perturb_method'].append(args.perturb_method)
        save_dic['accuracy'].append(correct/num_files)
        np.save(file_path, save_dic)

        path_wm_images = glob.glob(f'./watermarked_data/{args.model}/*/*')
        path_wm_perturbed_images = glob.glob(f'./perturbed_data2/{args.model}/*/*/*/*')
        save_dic = {}
        save_dic['temp'] = []
        save_dic['perturb_method'] = []
        save_dic['wm_method'] = []
        save_dic['psnr'] = []
        save_dic['ssim'] = []
        save_dic['bit_acc'] = []
        
        count = 0
        for i in tqdm(path_wm_images, desc = 'Calculating Metrics: '):
            wm_image_id = i.split('/')[-1]
            wm_fid_path = '/'.join(i.split('/')[:-1])
            
            for j in path_wm_perturbed_images:
                if wm_image_id in j:
                    split_wm_perturbed_imgs = j.split('/')
                    perturb_fid_path = '/'.join(j.split('/')[:-1])

                    save_dic['temp'].append(split_wm_perturbed_imgs[-2])
                    save_dic['perturb_method'].append(split_wm_perturbed_imgs[-3])
                    save_dic['wm_method'].append(split_wm_perturbed_imgs[-4])


                    save_dic['ssim'].append(ssim(np.array(Image.open(i)), np.array(Image.open(j)))[0])

                    save_dic['psnr'].append(psnr(np.array(Image.open(i)), np.array(Image.open(j))))

                    # BIT ACC # 
                    image_bits1, image_bits2 = np.array(Image.open(i), dtype=np.uint8), np.array(Image.open(j) ,dtype=np.uint8)
                    bit_accuracy = np.sum(image_bits1 == image_bits2) / image_bits1.size
                    save_dic['bit_acc'].append(bit_accuracy)

        df = pd.DataFrame(save_dic)
        average_ssim = df.groupby(['wm_method', 'perturb_method'])['ssim'].mean().reset_index()
        average_psnr = df.groupby(['wm_method', 'perturb_method'])['psnr'].mean().reset_index()
        average_bit = df.groupby(['wm_method', 'perturb_method'])['bit_acc'].mean().reset_index()
        print('SSIM: ', average_ssim)
        print('PSNR: ', average_psnr)
        print('bit acc: ', average_bit)
        save_dic = np.load(f'./detect_wm_image_{args.model}_2.npy', allow_pickle = True).item() 
        wm_method = save_dic['wm_method']
        perturb_method = save_dic['perturb_method']
        accuracy = save_dic['accuracy']
        temp = save_dic['temp']

        assert len(wm_method) == len(perturb_method) == len(accuracy) == len(temp)
        
        df = pd.DataFrame(save_dic)
        average_accuracy = df.groupby(['wm_method', 'perturb_method'])['accuracy'].mean().reset_index()
        print(average_accuracy)

        df.to_csv('metrics_bit_acc.csv', index=False)  

if __name__ == '__main__':
    args = get_args()
    main(args)