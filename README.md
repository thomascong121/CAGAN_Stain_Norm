# CAGAN_Stain_Norm

Configs
===
All configs are located `configs` folder

Training
====
For training the modeld section, simply run `python3 main.py dataset=cam16`. You may specifiy the dataset using `dataset=[tcga|breakhis|cam16|cam17]`.

Testing
====
For testing the model, simply run `python3 main.py dataset=cam16 run=test`. You may specifiy the dataset using `dataset=[tcga|breakhis|cam16|cam17]`.
For classifier training, simply run `python3 classifier.py dataset=cam17 model=classifier run.opt_run.n_epoch=40`. You may specifiy the dataset using `dataset=[tcga|breakhis|cam17]`.

Trained Packages
====
We provide several pretrained trained models.
We also provide our train-test [splits](https://drive.google.com/drive/folders/1eveV0teX4GGzUd0C3U8llRrdbhOv2riU?usp=sharing).

| method | Dataset | url |
|-------------------|-------------------|--------------------|
| CAGAN | TCGA-IDH | [model](https://drive.google.com/drive/folders/1oAdcyQxCCYoTXjKm3gKdjpA9L-aHE9Is?usp=sharing) |
| CAGAN | BreakHis | [model](https://drive.google.com/drive/folders/12c4feW0YTQNAGHIp8DXZHy39DElYjHEi?usp=sharing) | 
| CAGAN | CAMELYON16| [model](https://drive.google.com/drive/folders/1TB22uRbxofNKbl6fRKM-t7nRUVWkpAOw?usp=sharing) | 

Reference
====
@article{cong2022colour,
  title={Colour adaptive generative networks for stain normalisation of histopathology images},
  author={Cong, Cong and Liu, Sidong and Di Ieva, Antonio and Pagnucco, Maurice and Berkovsky, Shlomo and Song, Yang},
  journal={Medical Image Analysis},
  pages={102580},
  year={2022},
  publisher={Elsevier}
}
