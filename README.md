# D2SFormer: Dual attention-dynamic bidirectional Transformer for semantic segmentation of urban remote sensing images
## Installation

conda env create -n esst python=3.8
conda install pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=11.6.1
### copy from mmsegmentation==1.1.0
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .


### Train on Potsdam and Vaihingen dataset

 For example, when dataset is Potsdam and method is D2SFormer, you can run

python tools/train.py \
  --config configs/D2SFormer/D2SFormer_potsdam.py\
  --work-dir result/D2SFormer \
  --load_from path/to/pre-trained/model \

### Inference on Potsdam, Vaihingen and LoveDA dataset

For example, when dataset is Potsdam and method is D2SFormer, you can run

python tools/test.py \
  --config configs/D2SFormer/D2SFormer_potsdam.py \
  --checkpoint path/to/D2SFormer/model \
  --show_dir result/D2SFormer/test \

## Hyperparameters Configuration

Detailed hyperparameters config can be found in configs/_base_/

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

Method    Repository
Swin Transformer    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
UPerNet    https://github.com/CSAILVision/unifiedparsing
CSwin    https://github.com/microsoft/CSWin-Transformer

Cite this repository
If you use this software in your work, please cite it using the following metadata. Yi Yan, Jiafeng Li, Jing Zhang, Liuqian Wang, and Li Zhuo. (2024). D2SFormer by BJUT-AI&VBD [Computer software]. https://github.com/BJUT-AIVBD/D2SFormer
