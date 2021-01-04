# MLFinal
- Machine Learning Final Project
```
.
├── checkpoints
├── DataProcess
│   ├── datasets
│   ├── datasets.py
│   ├── data_split.py
│   ├── __init__.py
│   ├── mean_std.py
├── data_vision.py
├── model
│   ├── __init__.py
│   ├── mobilenetv1.py
│   ├── resnet.py
│   └── test.py
├── model_vision.ipynb
├── optimizer
│   ├── __init__.py
│   ├── pDCAe_exp.py
│   ├── pDCAe_nobeta.py
│   ├── SGD_capped.py
│   ├── SGD_l12_freeze.py
│   ├── SGD_l12.py
│   ├── SGD_l1.py
│   ├── SGD_mcp.py
│   └── SGD_SCAD.py
├── README.md
├── run.py
│── utils.py
├── run_mobilenetv1.sh
├── run_resnet18.sh
├── run_resnet50.sh
├── visualization.ipynb
├── multi_adaboost.py
├── data_vision.py
└── utils.py
```

# Requirements
 - torch:1.4.0 or higher
 - torchvision: 0.5.0 or higher
# Data Preprocess
If you download data from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), the first step you need to do is split data. You can run data_split.py
# Running Code
## Run some examples
The scripts of running all non-convex experiments are provided in **.sh** file. 
If you in the environment of RIIS's cluster, you can simply run the command to test 
```
bash run_mobilenetv1.sh
```

## Run specific experiment
- optimizer:[pDCAe_exp | pDCAe_nobeta | SGD_capped | SGD_l1 | SGD_l12 | SGD_l12_freeze | SGD_mcp | SGD_SCAD]
- model: [mobilenetv1 | resnet18 | resnet50 | resnet101]
- dataset_name: [chestX]

An example is:
```
python run.py --optimizer SGD_l1 --model mobilenetv1 --lambda_ 0.0001 --max_epoch 200 -lr 0.1 --batch_size 128
```

And there are some special parameter would be used in different algorithms.

# Reference
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
- Zhang, Yangjing, Kim-Chuan Toh, and Defeng Sun. "Learning Graph Laplacian with MCP." arXiv preprint arXiv:2010.11559 (2020).
- Le Thi, Hoai An, et al. "DC approximation approaches for sparse optimization." European Journal of Operational Research 244.1 (2015): 26-46.
- Wen, Bo, Xiaojun Chen, and Ting Kei Pong. "A proximal difference-of-convex algorithm with extrapolation." Computational optimization and applications 69.2 (2018): 297-324.