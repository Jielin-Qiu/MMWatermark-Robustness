# MMWatermark_Robustness

The official codebase for our paper "Evaluating Durability: Benchmark Insights into Multimodal Watermarking".

Jielin Qiu*, William Jongwon Han*, Xuandong Zhao, Shangbang Long, Christos Faloutsos, Lei Li.

More details can be found on the [project webpage](https://mmwatermark-robustness.github.io/).


## Citation

If you feel our code or models helps in your research, kindly cite our paper:

```
@inproceedings{Qiu2024EvaluatingDB,
  title={Evaluating Durability: Benchmark Insights into Multimodal Watermarking},
  author={Jielin Qiu and William Han and Xuandong Zhao and Shangbang Long and Christos Faloutsos and Lei Li},
  journal={arXiv preprint arXiv:2406.03728},
  year={2024}
}
```

## Getting Started


We generally recommend the following pipeline:

1. Generate text and images utilizing multimodal models.
2. Watermark generated text and images.
3. Perturb watermarked text and images.
4. Detect perturbed, watermarked text and image.

We will now go a bit more in depthon how to do each step.


## Environments


In our study, we follow the existing codebases for comprehensive benchmarking.

We recommend creating separate environments for each multimodal model and watermarking method. All perturbations (Text and Image) can be done through one environment.

One thing to note is that some of the links are not in fact repositories but Hugging Face tutorials on how to utilize the models. For such models, we experienced that downloading the latest transformers version works well. However, if there are any errors utilizing multiple multimodal models with a singular environment, please feel free to create another environment.

We provide the link to all of the necessary repositorys for this project. Please carefully follow their environment settings and generate, watermark, perturb in separate environments. We thank all of the repositories as well for open sourcing their code. 


| Type             | Link                                                                 |
|------------------|----------------------------------------------------------------------|
| Multimodal Model | [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)                     |
| Multimodal Model | [RPG](https://github.com/YangLing0818/RPG-DiffusionMaster)           |
| Multimodal Model | [LCMs](https://github.com/luosiallen/latent-consistency-model)       |
| Multimodal Model | [Kandinsky](https://github.com/ai-forever/Kandinsky-2)               |
| Multimodal Model | [PIXART](https://github.com/PixArt-alpha/PixArt-alpha)               |
| Multimodal Model | [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)    |
| Multimodal Model | [DALLE3](https://platform.openai.com/docs/guides/images/usage)       |
| Multimodal Model | [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| Multimodal Model | [Fuyu-8B](https://huggingface.co/adept/fuyu-8b)                      |
| Multimodal Model | [InternLM-XComposer](https://huggingface.co/internlm/internlm-xcomposer-2-7b) |
| Multimodal Model | [InstructBLIP](https://huggingface.co/docs/transformers/main/en/model_doc/instructblip) |
| Multimodal Model | [LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) |
| Multimodal Model | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)                |
| Multimodal Model | [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl) |
| Multimodal Model | [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)                           |
| Watermark        | [KGW](https://github.com/jwkirchenbauer/lm-watermarking)             |
| Watermark        | [KTH](https://github.com/jthickstun/watermark)                       |
| Watermark        | [Blackbox](https://github.com/Kiode/Text_Watermark)                  |
| Watermark        | [Unigram](https://github.com/XuandongZhao/Unigram-Watermark)         |
| Watermark        | [DwtDctSvd](https://github.com/ShieldMnt/invisible-watermark)        |
| Watermark        | [RivaGAN](https://github.com/ShieldMnt/invisible-watermark)          |
| Watermark        | [SSL](https://github.com/facebookresearch/ssl_watermarking)          |
| Watermark        | [Stega Stamp](https://github.com/tancik/StegaStamp)                  |
| Image and Text Perturbations        | [MM_Robustness](https://github.com/Jason-Qiu/MM_Robustness)                  |




## COCO Dataset

Please download the COCO validation split from the official website [cocodataset](https://cocodataset.org/#home). You can download [images-val2017](http://images.cocodataset.org/zips/val2017.zip) and [annotations-val2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) 

If for some reason there is a problem with the link, a copy of the data can be found [here](https://drive.google.com/drive/folders/1DFl0xkPkkQshoTk-81ksQ-XNJRRrJVKy?usp=sharing).

Then move the data into the COCO folder. the `coco.py` file is the data loader used to iterate through the data.


## Multimodal Models and Generation

All multimodal models used in this study is available in the `mm_model` directory. 
We do want to note that not all models had a Github repository, however, we still provide an example of how to utilize the model for text or image generation.

Additionally, we want to note that some of the models on Hugging Face are fairly large.
We recommend to set the model download cache path to a specific folder on your local machine that has enough memory. 

## Watermark

All watermarks are in the `watermark` directory. After setting up their respective enironments and having already generated the text or images, please proceed to watermark all generated texts or images.

## Perturbation

All perturbations are in the `perturbation` directory. After setting up the perturbation evironment from the `MM_Robustness` repository, please proceed to perturb all of the watermarked images or text. Additionally, inside the `perturbation` directory, the `image_perturb.py` and `text_perturb.py` files contain all of the needed image and text perturbations for this study. 

## Detection
Due to each watermarking method having their own way of detection we provide an example pipeline of detecting watermarks. Please view them to see examples of how to detect them. We also provide the calculation of the other metrics as well (e.g., ROUGE, PSNR, etc.).

## License

This project is licensed under CC BY-NC-SA License.

## Contact
If you have any questions, please contact wjhan@andrew.cmu.edu, jielinq@andrew.cmu.edu.
