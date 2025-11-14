# DeepLearning Course - Reproducibility Project (Facial Expression Recognition)
<img width="6699" height="1836" alt="image" src="https://github.com/user-attachments/assets/cf37265b-23f3-4ac8-8b00-0ac45839e069" />
This repository aims to reproduce and analyze the [POSTER V2: A simpler and stronger facial expression recognition network](https://arxiv.org/pdf/2301.12149)

The original paper introduces a **Patch Attention Transformer** architecture designed to capture both global and local facial features efficiently.  

In this project, we focus on verifying the reproducibility of POSTER V2 by retraining it on publicly available emotion datasets (AI Hub and custom VR data) and comparing results under different training conditions such as data augmentation and hyperparameter tuning.


## Folder Structure
```bash
├── checkpoint/ # Saved model weights and experiment logs
├── data/ # Original and processed datasets
│ ├── AIhub_train/
│ ├── AIhub_test/
├── data_preprocessing/ # Data cleaning, augmentation, and preparation scripts
├── log/ # Training logs, loss/accuracy curves, tensorboard outputs
├── models/ # Model definitions, training utilities
├── main.py # Primary training & evaluation pipeline
├── main_no_augmentation.py # Baseline training script without data augmentation
└── requirements.txt # Dependency list
```


## Datasets
Since the original dataset from the referenced paper was unavailable, we utilized:
* 1. [AI Hub Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82)
  * Composite Images for Korean Emotion Recognition
* 2. Custom VR headset user dataset (three classes: positive, neutral, negative)
  * We collected images of the lower frontal face of a user wearing a VR headset using an RGB camera. Due to privacy concern, these datasets are not publicly shareable.


## Model Architecture
The core model is based on a two-stream design that fuses **facial-landmark geometry** and **image-texture features** via cross-fusion transformer blocks. A Pyramid Structure allows the model to process features at multiple scales, providing a fine balance between context and detail.
Key strengths:
* Handles inter-class similarity (e.g., anger vs. sadness)
* Manages intra-class variation (age, ethnicity, facial structure)
* Operates robustly across scales and resolutions


## Installatioin
Install the project dependencies via pip:
```bash
pip install -r requirements.txt
```
* Ensure your environment meets the required versions of PyTorch, torchvision, scikit-learn, matplotlib, etc.


## How to Train & Evaluate
#### Training
```bash
python main.py --data path/to/dataset --lr 3.5e-5 --batch-size 64 --epochs 100 --gpu 0
```
* Training time: approx. 4 hours per epoch (on NVIDIA Tesla V100, batch size 32)
#### Evaluation / Inference
```bash
python main.py --data path/to/dataset --evaluate path/to/checkpoint
```
* Inference time: ~2 seconds per image


## Checkpoints
Pre-trained weights for key experiments are available: [Checkpoints](https://github.com/YewonMin/DeepLearning_Reproducibility/blob/main/checkpoint/save_ckp_here.txt)
| **Train Dataset**                | **Top-1 Accuracy** |**Path**       |
|----------------------------------|--------------------|---------------|
| AI Hub                           | 100                | checkpoint/try1_aihub_model_best.pth |
| Custom data                      | 81.40              | checkpoint/try3_ours_best.pth |
| Custom data (with augmentation)  | 92.81              | checkpoint/try4_ours_best.pth |
| Custom data (parameter search)   | 80.77              | checkpoint/try5_ours_best.pth |


## Reference
* Github: [POSTER_V2](https://github.com/Talented-Q/POSTER_V2)
* Paper: [POSTER_V2](https://www.sciencedirect.com/science/article/pii/S0031320324007027)
```bash
@article{mao2023poster, title={POSTER V2: A simpler and stronger facial expression recognition network}, author={Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin}, journal={arXiv preprint arXiv:2301.12149}, year={2023}}
```


## Conclusion
- This project successfully **reproduced the POSTER V2 facial expression recognition model**.  
- **Reproducibility** was validated on both AI Hub and custom VR datasets.  
- **Data augmentation** method significantly improve model performance.  
- Largely consistent results with the original paper demonstrate that POSTER V2 is **robust and generalizable**.  
