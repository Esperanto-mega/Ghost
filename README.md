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
### Dataset
We use three datasets in our paper, all of which have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6).

**NOTE:** We use the same model architecture as the backbone to generate item feature embeddings.

---
### Reproduction
1. Head-Tail preprocessing: Split the items into head and tail groups using `Ghost/data/process/process_head-tail.py`:
```bash
cd data/process
python process_head-tail.py
cd ../..
```
2. SKT: Run our SKT-based modified RQ-KMeans algorithm to generate SIDs. Set `--data_path` to the item embedding file in `Ghost/RQ-KMeans/run.sh`, then run: 
```bash
cd RQ-KMeans
bash run.sh
cd ..
```
3. Pretraining: Set `BASE_MODEL` to the pretrained language model in `Ghost/stage_1.sh`, then run: 
```bash
bash stage_1.sh
```
4. AUO: Set `BASE_MODEL` to the pretrained Ghost framework in `Ghost/stage_2.sh`, then run: 
```bash
bash stage_2.sh
```
5. Evaluation: Set `BASE_MODEL` and `CKPT_PATH` in `run_test_ddp.sh` to the base model and fine-tuned checkpoint paths, respectively, then run:
```bash
bash run_test_ddp.sh
```

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
