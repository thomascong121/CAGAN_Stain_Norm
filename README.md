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
We also provide our train-test [splits](https://drive.google.com/drive/folders/14eIMNuP2hqJKkJWF_McMKhsbzQpLmEj0?usp=sharing).

| method | Dataset | url |
|-------------------|-------------------|--------------------|
| CAGAN | TCGA-IDH | [model](https://drive.google.com/file/d/1XN-jyzkBCiYMGUYNHMj3hwusx6ROwh_G/view?usp=sharing) |
| CAGAN | BreakHis | [model](https://drive.google.com/file/d/1XN-jyzkBCiYMGUYNHMj3hwusx6ROwh_G/view?usp=sharing) | 
| CAGAN | CAMELYON16| [model](https://drive.google.com/file/d/1XN-jyzkBCiYMGUYNHMj3hwusx6ROwh_G/view?usp=sharing) | 

Reference
====
