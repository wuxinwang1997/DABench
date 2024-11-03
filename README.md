# DABench

The code repository for the manuscript [A Benchmark Dataset for AI-based Weather Data Assimilation](https://arxiv.org/abs/2408.11438)

## Abstract

Recent advancements in Artificial Intelligence (AI) have led to the development of several Large Weather Models (LWMs) that rival State-Of-The-Art (SOTA) Numerical Weather Prediction (NWP) systems. Until now, these models have still relied on traditional NWP-generated analysis fields as input and are far from autonomous. Currently, scientists are increasingly focusing on developing data-driven data assimilation (DA) models for LWMs. To expedite advancements in this field and facilitate the operationalization of data-driven end-to-end weather forecasting systems, we propose DABench, a benchmark constructed by simulated observations, real-world observations, and ERA5 reanalysis. DABench contributes four standard features: (1) sparse and noisy observations provided for both simulated and real-world experiments; (2) a Skillful pre-trained Transformer-based weather prediction model, Sformer, designed to generate background fields while rigorously assessing the impact of assimilation outcomes on predictions; (3) standardized evaluation metrics for the model comparison; (4) a strong DA baseline, 4DVarFormerV2. Our experimental results demonstrate that the end-to-end weather forecasting system, integrating 4DVarFormerV2 and Sformer, can assimilate real-world observations, thereby facilitating a stable DA cycle lasting one year and achieving a skillful forecasting lead time of up to 7 days. The proposed DABench will significantly advance research in AI-based DA, AI-based weather forecasting, and related domains.

## Datasets and Pretrained Models

We also provide datasets and pretrained models for the convenience of users.

- [DABench Dataset](https://pan.baidu.com/s/1P-omwjo-8tW8BMzH3QZklw)

## Baselines 

Untill now, we have benchmarked the following modes in this repo:
- [x] **SwinTransformer** [[Code]](https://github.com/ChristophReich1996/Swin-Transformer-V2)
- [x] **4DVarNet** - Learning Variational Data Assimilation Models and Solvers [[JAMES 2021]](https://onlinelibrary.wiley.com/doi/10.1029/2021MS002572) [[Code]](https://github.com/CIA-Oceanix/4dvarnet-core)
- [x] **4DVarFormer** . Accurate initial field estimation for weather forecasting with a variational-constrained neural network [[npj Climate and Atmospheric Science 2024](https://doi.org/10.1038/s41612-024-00776-1)] [[Code]]([./src/models/assimilate/dat/arch.py](https://github.com/wuxinwang1997/4DVarFormer))
- [x] **4DVarFormerV2** Proposed in this paper. (Will be uploaded [here](/src/models/assimilate/fdvarformerv2) after peer review.) 


## Citation

If you find our work useful in your research, please consider citing: -->

```
@misc{wang2024dabenchbenchmarkdatasetdatadriven,
  title={A Benchmark Dataset for AI-based Weather Data Assimilation}, 
  author={Wuxin Wang and Weicheng Ni and Tao Han and Taikang Yuan and Xiaoyong Li and Boheng Duan and Lei Bai and Kaijun Ren},
  year={2024},
  eprint={2408.11438},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2408.11438}, 
}
```