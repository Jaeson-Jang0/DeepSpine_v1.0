#!/bin/sh
#SBATCH --time=90:00:00
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --account=carney-tserre-condo
#SBATCH --partition gpu 
#SBATCH --gres=gpu:1

#SBATCH -J DeepSpine

#SBATCH --mail-user=jaeson_jang@brown.edu 
#SBATCH --mail-type=ALL

##### 20240312
python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v4_inferv230920.json
python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_sizematch.json
