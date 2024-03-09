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

##### 240209
python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C2_sizematch.json

##### 230207
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C2.json
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C3.json

##### 230828
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6.json

##### 230824
# python train_230703.py --config config/MLP_230817_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/MLP_230817_stimSling_from0_atOnset_paran6_n1.json
# python train_230703.py --config config/MLP_230817_stimSling_from0_atOnset_paran6_n2.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6_n2.json

##### 230823
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran4.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran2.json

##### 230821
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_gradual.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_preOnset.json

##### 230810
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from250.json

##### 230807
# python train_230703.py --config config/DeepSpine_230807_Paran8_left_DS1_mnsFB_bothTarget050.json

##### 230802
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget100.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget075.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget050.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget025.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget000.json

##### 230727
# python train_230703.py --config config/MLP_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS2.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS1_mnsFB.json

##### 230725
# python train_230703.py --config config/DeepSpine_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS3.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS4.json