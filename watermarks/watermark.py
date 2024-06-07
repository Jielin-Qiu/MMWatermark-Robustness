import glob
import os
import argparse
import numpy as np
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder
from tqdm import tqdm
import torch
torch.set_num_threads(2)
import re
from transformers import LogitsProcessorList
import time
from gptwm import GPTWatermarkLogitsWarper

from model.anyToImageVideoAudio import NextGPTModel
from config import *
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from mersenne import mersenne_rng
from generate import generate_shift

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--modality', type = str, default = None, help='Please choose the modality to watermark')
    parser.add_argument('--device', type = str, default = 'cuda:0', help = 'Please choose the type of device' )
    parser.add_argument('--method', type = str, default = None, help = 'Please choose the type of watermark method. For text, here are the options = [lm, robust]' )
    parser.add_argument("--model_name", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--wm_key", type=int, default=0)
    parser.add_argument("--prompt_file", type=str, default="./data/LFQA/inputs.jsonl")
    parser.add_argument("--output_dir", type=str, default="./data/LFQA/")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def main(args):
    # path_to_modality = './NExT-GPT/code/assets/'
    path_to_modality = './assets/'
    
    if args.modality == 'text':
        generated_text = np.load(os.path.join(path_to_modality, 'text/gen_captions.npy'), allow_pickle = True).item()
        watermarked_text_dic = {}

        assert len(generated_text.keys()) == len(set(list(generated_text.keys())))

        next_args = {'model': 'nextgpt',
                'nextgpt_ckpt_path': '../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/',
                'max_length': 128,
                'stage': 3,
                'root_dir': '../',
                'mode': 'validate',
                }
        
        next_args.update(load_config(next_args))

        model = NextGPTModel(**next_args)
        delta_ckpt = torch.load(os.path.join(next_args['nextgpt_ckpt_path'], 'pytorch_model.pt'), map_location=args.device)
        model.load_state_dict(delta_ckpt, strict=False)
        model = model.eval().cuda()
        tokenizer = model.llama_tokenizer
        llama_model = model.llama_model
        
        if args.method == 'lm':
            watermark_processor = WatermarkLogitsProcessor(vocab = list(tokenizer.get_vocab().values()),
                                                        gamma = 0.25, delta = 2.0, seeding_scheme = 'simple_1')
            

        elif args.method == 'unigram':
            watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=args.fraction,
                                                            strength=args.strength,
                                                            vocab_size=llama_model.config.vocab_size,
                                                            watermark_key=args.wm_key)])
            
        
        for key in tqdm(generated_text.keys(), desc = 'watermarking text...'):
            text = generated_text[key][0]
            if isinstance(text, dict):
                
                pass
            else:
                hash_position = text.find("#")
                if hash_position != -1:
                    extracted_text = text[:hash_position]
                else:
                    extracted_text = text
            
                if args.method=='lm':
                    tokenized_input = tokenizer(extracted_text, return_tensors = 'pt')
                    
                    output_tokens = llama_model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]))
                    
                    watermarked_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                    watermarked_text_dic[key] = watermarked_text

                elif args.method == 'unigram':
                    tokenized_input = tokenizer(extracted_text, return_tensors = 'pt')
                    num_tokens = len(tokenized_input['input_ids'][0])
                    generate_args = {
                            **tokenized_input.to(args.device),
                            'logits_processor': watermark_processor,
                            'output_scores': True,
                            'return_dict_in_generate': True,
                            'max_new_tokens': args.max_new_tokens,
                        }
                    generate_args['do_sample'] = True
                    generate_args['top_k'] = args.top_k
                    generate_args['top_p'] = args.top_p
                    
                    print('starting generate')
                    output_tokens = llama_model.generate(**generate_args)
                    
                    gen_text = tokenizer.batch_decode(output_tokens['sequences'], skip_special_tokens=True)
                    gen_text2 = tokenizer.batch_decode(output_tokens['sequences'][:, num_tokens: ], skip_special_tokens=True)
                    

                    watermarked_text_dic[key] = {'watermarked_text_all' : gen_text,
                                                 'watermarked_text_seg' : gen_text2}
                    
                elif args.method == 'robust':
                    
                    tokenized_start_time = time.time() 
                    tokens = tokenizer.encode(extracted_text, return_tensors = 'pt', truncation = True, max_length = 2048)
                    tokenized_end_time = time.time()
                    tokenized_time = tokenized_end_time - tokenized_start_time 
                    print(f"tokenized time: {tokenized_time} seconds")
                    watermark_start_time = time.time() 
                    watermarked_tokens = generate_shift(llama_model,tokens,len(tokenizer),256,30,42)[0]
                    watermarking_end_time = time.time()
                    watermark_time = watermarking_end_time - watermark_start_time 
                    print(f"watermark time: {watermark_time} seconds")

                    decode_start_time = time.time()
                    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
                    decode_end_time = time.time()
                    decode_time = decode_end_time - decode_start_time 
                    print(f"decode time: {decode_time} seconds")
                    watermarked_text_dic[key] = watermarked_text
            
        
        np.save(f'../../watermarked_data/text/{args.method}_watermarked_text.npy', watermarked_text_dic)

    elif args.modality == 'image':
        
        generated_image = glob.glob(os.path.join(path_to_modality, 'images/*'))
        
        ensure_directory_exists(f'./watermarked_data/image/{args.method}')

        encoder = WatermarkEncoder()
        if args.method == 'rivaGan':
            encoder.loadModel()
        wm = 'test'
        encoder.set_watermark('bytes', wm.encode('utf-8'))
    
        
        for path in tqdm(generated_image, desc = f'Watermarking Images With {args.method}: ' ):
            file_name = path.split('/')[-1]
            bgr = cv2.imread(path)
            bgr_encoded = encoder.encode(bgr, args.method)

            cv2.imwrite(f'./watermarked_data/image/{args.method}/{file_name}', bgr_encoded)

if __name__ == '__main__':
    args = get_args()
    main(args)