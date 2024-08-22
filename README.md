# DABench

The code repository for the manuscript submitted to NeurIPS 2024.

[DABench: A Benchmark Dataset for Data-Driven Weather Data Assimilation](https://arxiv.org/abs/2408.11438)

## Abstract

Recent advancements in deep learning (DL) have led to the development of several Large Weather Models (LWMs) that rival state-of-the-art (SOTA) numerical weather prediction (NWP) systems. Up to now, these models still rely on traditional NWP-generated analysis fields as input and are far from being an autonomous system. While researchers are exploring data-driven data assimilation (DA) models to generate accurate initial fields for LWMs, the lack of a standard benchmark impedes the fair evaluation among different data-driven DA algorithms. Here, we introduce DABench, a benchmark dataset utilizing ERA5 data as ground truth to guide the development of end-to-end data-driven weather prediction systems. DABench contributes four standard features: (1) sparse and noisy simulated observations under the guidance of the observing system simulation experiment method; (2) a skillful pre-trained weather prediction model to generate background fields while fairly evaluating the impact of assimilation outcomes on predictions; (3) standardized evaluation metrics for model comparison; (4) a strong baseline called the DA Transformer (DaT). DaT integrates the four-dimensional variational DA prior knowledge into the Transformer model and outperforms the SOTA in physical state reconstruction, named 4DVarNet. Furthermore, we exemplify the development of an end-to-end data-driven weather prediction system by integrating DaT with the prediction model. Researchers can leverage DABench to develop their models and compare performance against established baselines, which will benefit the future advancements of data-driven weather prediction systems. The code is available on [this Github repo](https://github.com/wuxinwang1997/DABench) and the dataset is available at [Baidu Drive](https://pan.baidu.com/s/1P-omwjo-8tW8BMzH3QZklw).

## Datasets and Pretrained Models

We also provide datasets and pretrained models for the convenience of users.

- [DABench Dataset](https://pan.baidu.com/s/1P-omwjo-8tW8BMzH3QZklw)
<!-- - [DaT Pretrained Model](https://pan.baidu.com/s/1P-omwjo-8tW8BMzH3QZklw) -->

## Leaderboard of DABench

Untill now, we have benchmarked the following modes in this repo:
- [x] **4DVarNet** - Learning Variational Data Assimilation Models and Solvers [[JAMES 2021]](https://onlinelibrary.wiley.com/doi/10.1029/2021MS002572) [[Code]](https://github.com/CIA-Oceanix/4dvarnet-core)
<!-- - [ ] **Adas** - Towards an end-to-end artificial intelligence driven global weather forecasting system [[Arxiv 2024]](http://arxiv.org/abs/2312.12462v3) [[Code]](./src/models/assimilate/adas/arch.py)
- [ ] **Fuxi-DA** - Fuxi-DA: A Generalized Deep Learning Data Assimilation Framework for Assimilating Satellite Observations [[Arxiv 2024]](http://arxiv.org/abs/2404.08522) [[Code]](./src/models/assimilate/fuxi_da/arch.py) -->
- [x] **SwinTransformer** [[Code]](https://github.com/ChristophReich1996/Swin-Transformer-V2)
- [x] **DaT** Proposed in this paper. [[Code]](./src/models/assimilate/dat/arch.py)

<!-- ## Benchmarking results 

### One-year data assimilation cycle
The first set of results reported the RMSEs and ACCs at 4 different variables: z500, t850, q700, u850.


### 15-day medium-range forecast
The first set of results reported the RMSEs and ACCs at 4 different variables: z500, t850, q700, u850.
-->

## Citation

If you find our work useful in your research, please consider citing: -->

```
@misc{wang2024dabenchbenchmarkdatasetdatadriven,
  title={DABench: A Benchmark Dataset for Data-Driven Weather Data Assimilation}, 
  author={Wuxin Wang and Weicheng Ni and Tao Han and Lei Bai and Boheng Duan and Kaijun Ren},
  year={2024},
  eprint={2408.11438},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2408.11438}, 
}
```