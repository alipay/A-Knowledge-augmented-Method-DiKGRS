# A-Knowledge-augmented-Method-DiKGRS
This is our PyTorch implementation for the paper: "*Towards Improving Trustworthiness of Personalized Online Service: A Knowledge-augmented Method*". The paper is under-review and the ArXiv version will release soon.

## Environment Requirement
The code has been tested running under Python 3.7.13. The required packages are as follows:
* torch == 1.7.1+cu110
* numpy == 1.21.5
* gensim == 4.2.0
* networkx == 2.6.3
* pandas == 1.0.0

## Example to Run the Codes
* First, make directions for log files:
```bash
mkdir Logs
```
* For DiKGRS with VRKG backbone on last-fm dataset, run:
```bash
python main_vrkg_kgin.py --log_fname 'Last_PSDVN_2k_ni_2.txt' --dataset last-fm --add_pseudo True --add_dvn True --fusion_gate True --batch_size 1024 --dim 256 --n_iter 2 --num_ps 1846 --random_seed 2020
```

* For DiKGRS with KGIN backbone on movie dataset, run:
```bash
python main.py --log_fname 'KGIN_movie_PSDVN_2k_ni_3_64.txt' --dataset movie --add_pseudo True --backbone KGIN --add_dvn True --fusion_gate True --batch_size 1024 --dim 64  --num_ps 2347 --random_seed 2020
```
* For DiKGRS with CKE backbone on movie dataset, run:
```bash
python main_cke.py --dataset movie --add_dvn True --add_pseudo True --lr 0.001 --cf_l2loss_lambda 0.005 --kg_l2loss_lambda 0.005 --num_ps 1374
```
* For DiKGRS with KGAT backbone on last-fm dataset, run:
```bash
python main_kgat.py --dataset last-fm --add_dvn True --add_pseudo True --num_ps 1846 --lr 0.001 --cf_l2loss_lambda 0.0005 --kg_l2loss_lambda 0.0005
```

* We have provided the generated pseudo-samples in the text files, with file paths: data/[dataset name]/pseudo_ratings.txt. To reproduce the pseudo_ratings files, you can run the code as follows.
```bash
python pseudo_entry_generation.py --dataset [dataset_name]
```
The hyperparameters can refer to the supplementary file of our paper and can be specified in parsers.
 
