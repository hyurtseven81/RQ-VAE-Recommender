# RQ-VAE Recommender — Level-Aware Hybrid Decoding

> **Research fork.** This extends the original TIGER implementation with level-aware hybrid decoding for generative retrieval. See `paper/` for the CIKM 2026 submission draft. Implementation plan: `docs/implementation_plan.md`.

This is a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval". 
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE (figure below).
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based is trained on sequences of semantic IDs to generate the next ids in the sequence.
![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)

### Currently supports
* **Datasets:** Amazon Reviews (Beauty, Sports, Toys, Steam), MovieLens 1M, MovieLens 32M
* RQ-VAE Pytorch model implementation + KMeans initialization + RQ-VAE Training script.
* Decoder-only retrieval model + Training code with semantic id user sequences from randomly initialized or pretrained RQ-VAE.
* **Level-aware hybrid decoding** (`modules/decoding/`): per-codebook-level alpha mixing of AR and dense scores during beam expansion — DBS, Gumbel, Hybrid, SASRec rerank, grid/learned alpha variants.
* **SASRec auxiliary head** (`modules/heads/sasrec_head.py`): 384→32 MLP projecting decoder states to item space; initialized from RQ-VAE encoder outputs.
* **Multi-task training** (`train_decoder_mtl.py`): joint SID + InfoNCE auxiliary loss with lambda warmup.
* **Evaluation harness** (`evaluate/`): Recall@K, NDCG@K, per-user tracking, paired bootstrap significance tests.
* **Residual entropy analysis** (`modules/analysis/residual_entropy.py`): validates coarse-to-fine codebook structure.

#### Level-aware decoding — quick start
```bash
# Grid-search alpha schedule (no training needed)
python scripts/alpha_grid_search.py --dataset beauty --decoder-ckpt <path> --rqvae-ckpt <path>

# Multi-task training (adds SASRec aux head)
python train_decoder_mtl.py configs/decoder_amazon_beauty_mtl.gin

# Evaluate with level-aware mixing
python evaluate/run_eval.py --strategy level_aware_mix_learned --dataset beauty
```

### 🤗 Usage on Hugging Face 
RQ-VAE trained model checkpoints are available on Hugging Face 🤗: 
* [**RQ-VAE Amazon Beauty**](https://huggingface.co/edobotta/rqvae-amazon-beauty) checkpoint.

### Installing
Clone the repository and run `pip install -r requirements.txt`. 

No manual dataset download is required.

### Executing
RQ_VAE tokenizer model and the retrieval model are trained separately, using two separate training scripts. 
#### Custom configs
Configs are handled using `gin-config`. 

The `train` functions defined under `train_rqvae.py` and `train_decoder.py` are decorated with `@gin.configurable`, which allows all their arguments to be specified with `.gin` files. These include most parameters one may want to experiment with (e.g. dataset, model sizes, output paths, training length). 

Sample configs for the `train.py` functions are provided under `configs/`. Configs are applied by passing the path to the desired config file as argument to the training command. 
#### Sample usage
To train both models on the **Amazon Reviews** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_amazon.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_amazon.gin`

To train both models on the **MovieLens 32M** dataset, run the following commands:
* **RQ-VAE tokenizer model training:** Trains the RQ-VAE tokenizer on the item corpus. Executed via `python train_rqvae.py configs/rqvae_ml32m.gin`
* **Retrieval model training:** Trains retrieval model using a frozen RQ-VAE: `python train_decoder.py configs/decoder_ml32m.gin`

### End-to-end pipeline

The full pipeline has four stages: data prep → RQ-VAE training → decoder training → evaluation. Run all steps from the repo root.

#### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install -e ".[dev,experiments]"   # optional: dev tools + experiment scripts
```

#### 2. Data prep
Amazon (Beauty, Sports, Toys) download automatically. Steam requires a one-time cache step:
```bash
python -c "from data.steam import RawSteam; RawSteam().download()"
```

#### 3. Train RQ-VAE tokenizer
```bash
# Amazon Beauty / Sports / Toys (set dataset= in gin config)
python train_rqvae.py configs/rqvae_amazon.gin

# Steam
python train_rqvae.py configs/rqvae_steam.gin

# MovieLens 32M
python train_rqvae.py configs/rqvae_ml32m.gin
```
Checkpoints are saved under `trained_models/`.

#### 4. Train decoder

**Standard (SID cross-entropy only):**
```bash
python train_decoder.py configs/decoder_amazon.gin
```

**Multi-task (SID + SASRec InfoNCE auxiliary loss):**
```bash
python train_decoder_mtl.py configs/decoder_amazon_beauty_mtl.gin
# variants: decoder_amazon_sports_mtl.gin, decoder_amazon_toys_mtl.gin, decoder_steam_mtl.gin
```

#### 5. Evaluate with level-aware decoding

**Grid-search alpha schedule** (no additional training, tries 5^3=125 configs):
```bash
python scripts/alpha_grid_search.py \
  --dataset beauty \
  --decoder-ckpt trained_models/decoder_beauty/checkpoint.pt \
  --rqvae-ckpt trained_models/rqvae_amazon_beauty/checkpoint_399999.pt
```

**Learned alpha schedule** (fine-tunes AlphaParams for 1 epoch):
```bash
python scripts/train_alpha_params.py \
  --dataset beauty \
  --decoder-ckpt trained_models/decoder_beauty/checkpoint.pt \
  --rqvae-ckpt trained_models/rqvae_amazon_beauty/checkpoint_399999.pt
```

**Run evaluation** (all decoding strategies):
```bash
python evaluate/run_eval.py \
  --strategy level_aware_mix_learned \
  --dataset beauty \
  --decoder-ckpt trained_models/decoder_beauty/checkpoint.pt \
  --rqvae-ckpt trained_models/rqvae_amazon_beauty/checkpoint_399999.pt
# results saved to results/
```

#### 6. Generate paper figures and tables
```bash
python scripts/make_figures.py --results results/all_runs.parquet --output paper/figures/
python scripts/make_tables.py  --results results/all_runs.parquet --output paper/tables/
```

#### Running on SageMaker
All training stages have corresponding launch scripts under `sagemaker/launch/`. Upload datasets and checkpoints first:
```bash
# From a machine with YOUR_AWS_PROFILE AWS profile
python sagemaker/setup/upload_datasets.py
aws s3 cp trained_models/ s3://YOUR_S3_BUCKET/rqvae-level-aware/checkpoints/ --recursive --profile YOUR_AWS_PROFILE

# Launch training
python sagemaker/launch/launch_rqvae.py --dataset beauty
python sagemaker/launch/launch_mtl.py --dataset beauty
python sagemaker/launch/launch_decoding_eval.py --dataset beauty --strategy level_aware_mix_learned
```

### References
* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy
* [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) by Eric Jang, Shixiang Gu, Ben Poole
* [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) by Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, Ehsan Amid, Sebastian Thrun, Christopher Ré
* [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) by lucidrains
* [deep-vector-quantization](https://github.com/karpathy/deep-vector-quantization) by karpathy
  
