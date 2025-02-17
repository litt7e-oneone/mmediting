# 准备 SIDD 数据集

<!-- [DATASET] -->

```bibtex
@inproceedings{Zamir2021Restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={CVPR},
  year={2022}
}
```

训练数据集可以从 [此处](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/) 下载。验证数据集可以从 [此处](https://drive.google.com/file/d/1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ/) 下载。测试数据集可以从 [此处](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/) 下载。

测试数据集需要从 mat 文件中导出，为此，我们提供了一个脚本：

```shell
python tools/dataset_converters/sidd/preprocess_sidd_test_dataset.py --data-root ./data/SIDD/test --out-dir ./data/SIDD/test
```

文件目录结构应如下所示：

```text
mmediting
├── mmedit
├── tools
├── configs
├── data
|   ├── SIDD
|   |   ├── train
|   |   |   ├── gt
|   |   |   ├── noisy
|   |   ├── val
|   |   |   ├── input_crops
|   |   |   ├── target_crops
|   |   ├── test
|   |   |   ├── gt
|   |   |   ├── noisy
```
