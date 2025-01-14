-  Env Install
```
conda create -n sam python==3.10 -y
conda activate sam
git clone git@github.com:huggingface/transformers.git && cd transformers && pip install -q .
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx opencv-python monai
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
