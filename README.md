# [LOTA: Bit-Planes Guided AI-Generated Image Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.html),&nbsp;&nbsp;   [ICCV25_Published](https://openaccess.thecvf.com//content/ICCV2025/papers/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.pdf), &nbsp;&nbsp;[arXiv](https://arxiv.org/abs/2510.14230)

**Contributors: Hongsong Wang (hongsongwang@seu.edu.cn), Renxi Cheng (renxi@seu.edu.cn), Jie Gui (guijie@seu.edu.cn)**

**Southeast University, Nanjing, China**

## 📰 News
- 🚨 **The whole ICCV 2025 papers with Codes are summarized on [ICCV2025_ABSTRACT/](https://hongsong-wang.github.io/ICCV2025_ABSTRACT/)**
- 🚨 **Paper Portal for Top Conferences in the Field of Artificial intelligence: [CV_Paper_Portal](https://hongsong-wang.github.io/CV_Paper_Portal/)**

## Hightlights
<p align="center">
  <img src="images/intro_00.png" width="55%">
<br>
  <b>Figure 1: Comparison of least bit-planes between real images and AI-generated images.</b>
</p>


<p align="center">
  <img src="images/method_00.png" width="70%">
<br>
  <b>Figure 2: Overview of our method.</b>
</p>

**Novel solution for AI-generated image detection**: We innovatively address AI-generated image detection based on bit-planes, and propose an efficient approach for noisy representation extraction. 

**Efficientpipeline design**: We propose a simple yet effec tive pipeline with three modules: noise generation, patch selection and classification. We design a heuristic strategy called maximum gradient patch selection and introduce two effective classifiers: noise-based classifier and noise guided classifier. Our approach operates at millisecond level, nearly a hundred times faster than current methods. 

**Exceedingly superior performance**: Extensive exper iments demonstrate the effectiveness of LOTA, which achieves 98.9% ACC onGenImage, showing great cross-generator generalization capability and outperforming existing mainstream methods by more than 11.9%.

## Dataset
We use [GenImage](https://github.com/GenImage-Dataset/GenImage) for training and evaluation, which can be downloaded online. GenImage is composed of 8 subsets (BigGAN, Midjourney, Wukong, Stable_Diffusion_v1.4, Stable_Diffusion_v1.5, ADM, GLIDE, VQDM), each of which contains fake images and real images from ImageNet. Additionally, each subset are invided into training dataset and validating dataset, and we train LOTA on the training dataset of one subset (e.g., Stable_Diffusion_v1.5) and evaluate on the validating dataset of all subsets.

## Training
You can train the LOTA model on Stable_Diffusion_v1.5 of GenImage by running the following command:
```
python train.py --choice=[0, 0, 0, 0, 1, 0, 0, 0]
                --image_root='Path/to/GenImage'
                --save_path='Path/to/saved_weights'
                --bit_mode='scaling'
                --patch_size=32
                --patch_mode='random'
```

## Evaluation
You can evaluate LOTA on all subsets of GenImage by running the following command:
```
python test.py  --choice=[1, 1, 1, 1, 1, 1, 1, 1]
                --load='Path/to/saved_weights'
                --image_root='Path/to/GenImage'
                --bit_mode='scaling'
                --patch_size=32
                --patch_mode='max'
```
Additionally, we provide the pretrained [weights](https://pan.baidu.com/s/1Eid3NYcsPC8ON7L90iM6Jg?pwd=srk3) (code: srk3) on Stable_Diffusion_v1.5 for evaluation. You can download the weights and easily evaluate LOTA on GenImage.

## Results
We provide the evaluation results in [Results](https://github.com/hongsong-wang/LOTA/tree/main/results), which contains path_to_testing_images, true_label, predict_prob_real, and predict_prob_fake of all testing images in GenImage.

## Acknowledgments
This repository borrows partially from [CNNDetection](https://github.com/PeterWang512/CNNDetection), [PatchCraft](https://github.com/cvlcgabriel/PatchCraft) and [SSP](https://github.com/bcmi/SSP-AI-Generated-Image-Detection). Thanks for their work sincerely.

```
@InProceedings{Wang_2025_ICCV,
    author    = {Wang, Hongsong and Cheng, Renxi and Zhang, Yang and Han, Chaolei and Gui, Jie},
    title     = {LOTA: Bit-Planes Guided AI-Generated Image Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {17246-17255}
}

