# Compositional_Crystal_Generation
This is the official implement of paper [Composable Crystals: Controllable Materials Discovery via Concept Learning](https://arxiv.org/pdf/2605.14769).

<img src="https://github.com/liun-online/Compositional_Crystal_Generation/blob/main/composition.png" width="800">

## Citation
```
@article{liu2026composable,
  title         = {Composable Crystals: Controllable Materials Discovery via Concept Learning},
  author        = {Liu, Nian and Zeng, Yuwei and Kubo, Ryoji and Kazeev, Nikita and Dale, Stephen Gregory and Maevskiy, Artem and Huang, Pengru and Laurent, Thomas and Novoselov, Kostya S. and Bresson, Xavier},
  journal       = {arXiv preprint},
  archivePrefix = {arXiv},
  eprint        = {2605.14769},
  year          = {2026}
}
```

## Python environment setup with Conda
```
conda create -n compo python=3.12.0
conda activate compo
conda install -c conda-forge mattersim==1.2.0
conda install -c conda-forge pymatgen=2025.6.14 ase=3.25.0 matminer=0.9.3
pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install p_tqdm lmdb easydict einops atomate2
pip install setuptools==80.9.0 faiss-cpu==1.12.0 lmdb==1.6.2 scipy==1.16.2 smact==3.2.0

git clone https://github.com/liun-online/Compositional_Crystal_Generation.git
cd Compositional_Crystal_Generation/
```

## Download and Reproduce
Firstly, install [Hugging Face](https://huggingface.co/) and login
```
pip install -U huggingface_hub
hf auth login
```
Then, download the folders, i.e., `./data`, `./exp_logs`, `./ref_dataset` and `exp_data.zip` by running the following commands:
```
hf download liun-online/Compositional_Crystal_Generation --repo-type dataset --local-dir ./
mv ./ref_dataset ./eval/vsun/
```
The `./exp_logs` folder and `exp_data.zip` file include the checkpoints and data for reproducing the results in paper.

To reproduce V.S.U.N.
- MP-20: 35.7
- Alex-MP-20: 43.7
```
mv exp_logs logs
unzip exp_data.zip
mv data data_clean
mv exp_data data

## MP-20
python XI_base_gen.py
python XII_base_eval.py

## Alex-MP-20
python XI_base_gen.py --dataset alex_mp_20
python XII_base_eval.py --dataset alex_mp_20

## [Optional] Run the commands below to prevent name clashes when starting a new training run
mv data exp_data
mv data_clean data
mv logs exp_logs
```
Remarks
> 1. `python XI_base_gen.py` generates 10 batches of 1,000 crystals, stored in `./logs/base/mp_20/gen` and `./logs/base/alex_mp_20/gen`
> 2. Find results at `./logs/base/mp_20/eval/metrics_summary.log` and `./logs/base/alex_mp_20/eval/metrics_summary.log`

## Training pipeline
### a. Train VQ-VAE and Extract Concepts
We follow Appendix B to train a VQ-VAE here.
```
python I_train_vae.py
python II_faiss_centroid.py
python III_train_vqvae.py
```

### b. Train a Composition Generatior and Refine it
1) Quantize the data using the extracted concepts, and learn these concept compositions
```
python IV_quantize_data.py
python V_train_compo_gen.py
```
2) Generate composition candidates, and keep the ones V.S.U.N relative to the training crystals
```
python VI_gen_compo.py
python VII_filter_vsun_compo.py
```
3) Refine the composition generator, and then generate final compositions to condition following base generative model
```
python VIII_ft_compo_gen.py
python IX_gen_compo_for_base.py
```

### c. Train a Composition-based Generative Model and Evaluate
```
python X_train_base.py
python XI_base_gen.py
python XII_base_eval.py
```
Remarks
> 1. Add `--dataset alex_mp_20` behind each command in stage b and c to use Alex-MP-20 dataset.
> 2. Change hyper-parameters, e.g., ```python V_train_compo_gen.py --conf_new training.batch_size=256  training.lr=0.0001```
> 3. Use partial of GPUs, e.g., ```CUDA_VISIBLE_DEVICES=0,1 V_train_compo_gen.py```. Default: use all GPUs.
> 4. The evaluation code is mainly adapted from [MatterGen](https://github.com/microsoft/mattergen) and [FlowMM](https://github.com/facebookresearch/flowmm).
