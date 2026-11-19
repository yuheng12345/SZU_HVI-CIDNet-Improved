# HVI-CIDNet 复现与局部自适应亮区保护

本项目基于 HVI-CIDNet 预训练模型，复现低光照图像增强，
并在推理结果之后加入局部自适应融合模块，尝试保护图像中的亮区层次。

## 原项目

- 论文：HVI: A New Color Space for Low-light Image Enhancement，CVPR 2025
- 官方代码：https://github.com/Fediory/HVI-CIDNet
- 原项目说明见 `UPSTREAM_README.md`
- 原项目许可证见 `LICENSE`；请在使用和发布前阅读许可证条款

## 文件说明

- `eval_hf.py`：原作者的单图推理脚本
- `scripts/eval_hf_adaptive_linear.py`：线性权重融合
- `scripts/eval_hf_adaptive.py`：Sigmoid 权重融合
- `scripts/batch_compare.py`：原图、原版与两种融合结果的对比图
- `net/`、`loss/`、`data/`：原项目相关代码

## 实验环境

已使用的环境：Linux、Python 3.7.16、PyTorch 1.13.1+cu117、
torchvision 0.14.1+cu117。依赖文件来自原项目；
实际安装时请结合 Python 和 CUDA 环境检查兼容性。

数据集、模型权重及实验输出不包含在本仓库中。
使用前需自行获取相应数据和作者预训练权重，并确认其许可。

## 方法说明

对输入图像运行 CIDNet 后，根据原图局部亮度和模型增亮幅度计算融合权重，
经过平滑后融合原图与增强图。该模块不修改 CIDNet 的预训练参数。
