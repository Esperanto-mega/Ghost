# Ghost
[Echoes in Filter Bubble: Diagnosing and Curing Popularity Bias in Generative Recommenders.](https://arxiv.org/pdf/2605.16825) (NeurIPS 2026)

<img width="1468" height="512" alt="QQ_1779762707970" src="https://github.com/user-attachments/assets/bdb38a11-58a0-4f78-935e-6a6485f7232f" />
<div align="center">
  <img width="1468" height="532" alt="image" src="https://github.com/user-attachments/assets/84e07383-4620-4e64-9b76-42611b5ee52c" />
</div>

### Environment Requirement
```
torch
transformers
deepspeed
peft
numpy
tqdm
polars
k-means-constrained
```

---
### Reproduction
1. Head-Tail preprocessing: Split the items into head and tail groups using `Ghost/data/process/process_head-tail.py`.
2. SKT: Run our SKT-based, modified RQ-KMeans algorithm to generate SIDs. The script is located at `Ghost/RQ-KMeans/run.sh`.
3. Pretraining: Pretrain the Ghost framework using `Ghost/stage_1.sh`.
4. AUO: Finetune the pretrained Ghost framework to mitigate popularity bias using `Ghost/stage_2.sh`.
5. Evaluation: Evaluate Ghost using `Ghost/run_test_ddp.sh`.

---
If this work is helpful, please cite this paper as the reference, through

```bibtex
@article{yin2026echoes,
  title={Echoes in Filter Bubble: Diagnosing and Curing Popularity Bias in Generative Recommenders},
  author={Yin, Jun and Zhu, Bangguo and Huo, Peng and Liu, Ruochen and Chen, Hao and Wang, Senzhang and Pan, Shirui and Zhang, Chengqi},
  journal={arXiv preprint arXiv:2605.16825},
  year={2026}
}
```
