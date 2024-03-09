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
python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_sizematch.json

##### 240207
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920.json
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C1.json
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C2.json
# python train_230703.py --config config/RNN_231027_treadmill_wEES2309_v4_inferv230920_C3.json

##### 231205
# python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v4_inferv230920_C4.json

##### 231204
# python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v4_inferv230920_forIllu.json

##### 231027
# python train_230703.py --config config/RNN_231027.json

# ##### 231025
# python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v4_inferv230920_C1.json

# ##### 231017
# python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v4_inferv230920_xy.json

# ##### 231010
# python train_230703.py --config config/DeepSpine_231010_treadmill_wEES2309_v3_inferv230920.json

# ##### 231006
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920_100ms.json
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_btwStim_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920_100ms.json

##### 231003
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920.json
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230928_999.json
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_btwStim_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230928_999.json
# python train_230703.py --config config/DeepSpine_231003_treadmill_wEES2309_btwStim_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920.json

##### 230929
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_B1_kineTarget100_inferv230920.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_B1_btwStim_kineTarget100_inferv230920.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_B1_2DS_kineTarget100_inferv230920.json

##### 230928
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230928_999.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_btwStim_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230928_999.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_2DS_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230928_999.json

# ##### 230927
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230927.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920.json

# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_btwStim_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_2DS_kineTarget100_mnsFB_atOnset_paran6__05_1_inferv230920.json

##### 230926
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget100_offEES_mnsFB_atOnset_paran6__05_1_infer.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_kineTarget050_mnsFB_atOnset_paran6__05_1_infer.json
# python train_230703.py --config config/DeepSpine_230926_treadmill_wEES2309_2DS_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

##### 230925
# python train_230703.py --config config/DeepSpine_230817_treadmill_nonEES2306_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

# python train_230703.py --config config/DeepSpine_230817_treadmill_withEES2208_4DS_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

# ##### 230921
# python train_230703.py --config config/DeepSpine_230817_treadmill_nonEES2208_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

# python train_230703.py --config config/DeepSpine_230817_treadmill_withEES2208_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

# python train_230703.py --config config/DeepSpine_230817_treadmill_withEES2208_nonEESvalid_kineTarget100_mnsFB_atOnset_paran6__05_1_infer.json

# python train_230703.py --config config/DeepSpine_230817_treadmill_nonEES2306_kineTarget050_mnsFB_atOnset_paran6__05_1_infer.json

##### 230912
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_infer__05_1.json

# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6__05_1.json
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6__025_1.json
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6.json

##### 230901
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_infer__025_1.json
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_infer__05_1.json

# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6__025_05.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6__025_1.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6__05_1.json

##### 230831
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6_n2__05_1.json
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6_n1__05_1.json
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_n2__05_1.json
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_n1__05_1.json

# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_infer__05_1.json
# python train_230703.py --config config/DeepSpine_230817_both_mnsFB_bothTarget050_stimSling_atOnset_paran6_infer__025_1.json

# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6__025_1.json
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6__05_1.json

##### 230828
# python train_230703.py --config config/MLP_230817_both_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_both_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6_infer.json


##### 230824
# python train_230703.py --config config/MLP_230817_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/MLP_230817_stimSling_from0_atOnset_paran6_n2.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6_n2.json

##### 230823
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran6.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran4.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset_paran2.json

##### 230817
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_preOnset.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_gradual.json
# python train_230703.py --config config/DeepSpine_230817_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0_atOnset.json

# 
##### 230809
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget050_stimSling_from0.json

##### 230807
# python train_230703.py --config config/DeepSpine_230807_Paran8_left_DS1_mnsFB_bothTarget050.json

##### 230802
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget100.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget075.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget050.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget000.json
# python train_230703.py --config config/DeepSpine_230802_Paran8_left_DS1_mnsFB_bothTarget025.json

##### 230731
# python train_230703.py --config config/DeepSpine_230731_Paran8_left_DS1_mnsFB_bothTarget100.json
# python train_230703.py --config config/DeepSpine_230731_Paran8_left_DS1_mnsFB_bothTarget075.json
# python train_230703.py --config config/DeepSpine_230731_Paran8_left_DS1_mnsFB_bothTarget050.json
# python train_230703.py --config config/DeepSpine_230731_Paran8_left_DS1_mnsFB_bothTarget025.json
# python train_230703.py --config config/DeepSpine_230731_Paran8_left_DS1_mnsFB_bothTarget000.json

##### 230728
# python train_230703.py --config config/DeepSpine_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS1_mnsFB.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS2_mnsFB.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS3_mnsFB.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS4_mnsFB.json

##### 230727
# python train_230703.py --config config/MLP_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS1.json
# python train_230703.py --config config/MLP_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS2.json
# python train_230703.py --config config/MLP_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS3.json
# python train_230703.py --config config/MLP_230725_Paran8_right_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS4.json

##### 230725
# python train_230703.py --config config/DeepSpine_230720_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_mnsFB.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS1.json
# python train_230703.py --config config/DeepSpine_230725_Paran8_left_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW_DS2.json

##### 230720
# python train_230703.py --config config/DeepSpine_230720_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_NBC_LongW.json

# python train_230703.py --config config/DeepSpine_230720_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_NBC_mnsFB.json
# python train_230703.py --config config/DeepSpine_230720_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_NBC_Kine.json

# python train_230703.py --config config/DeepSpine_230720_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_NBC.json

##### 230719
# python train_230703.py --config config/DeepSpine_230719_Paran16_leftknee_WBFclip_t25_unified_5_EMG_5s_mid.json
# python train_230703.py --config config/DeepSpine_230719_Paran16_leftknee_WBFclip_t25_unified_5_EMG_5s.json
# python train_230703.py --config config/DeepSpine_230718_Paran16_leftknee_WBFclip_t25_unified_5_EMG_5s_mid.json
# python train_230703.py --config config/DeepSpine_230718_Paran16_leftknee_WBFclip_t25_unified_5_EMG_5s.json

# python train_230703.py --config config/DeepSpine_230719_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s.json
# python train_230703.py --config config/DeepSpine_230719_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid.json

##### 230718
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_Lr10d.json
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid.json
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_Lr10d.json
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_Lr100d.json
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s_mid_Lr100d.json

# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG_5s.json
# python train_230703.py --config config/DeepSpine_230718_Paran8_leftknee_WBFclip_t25_unified_5_EMG.json

##### 230713
# python train_230703.py --config config/DeepSpine_230713_Paran8_leftknee_WBFclip_t25_unified_5_EMG.json
# python train_230703.py --config config/DeepSpine_230713_Paran8_leftknee_WBFclip_t25_unified_5_Kine.json

##### 230706
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified.json
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified_1.json
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified_2.json
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified_3.json
# python train_230703.py --config config/DeepSpine_230706_Paran1_leftknee_WBFclip_t25_unified_5.json
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified_4.json
# python train_230703.py --config config/DeepSpine_230706_Paran8_leftknee_WBFclip_t25_unified_5.json

##### 230705
# python train_230703.py --config config/DeepSpine_230705_Paran1_leftknee_WBFclip_KineEMG_unified.json

# 230703
# python train_230703.py --config config/DeepSpine_230703_Paran1_leftknee_WBFclip_unified.json
# python train_230703.py --config config/DeepSpine_230703_Paran2_leftknee_WBFclip_unified.json
# python train_230703.py --config config/DeepSpine_230703_Paran2_leftknee_WBFclip_wLN_unified.json
# python train_230703.py --config config/DeepSpine_230703_Paran8_leftknee_WBFclip_unified.json
# python train_230703.py --config config/DeepSpine_230703_Paran8_leftknee_WBFclip_wLN_unified.json

# python train_230703.py --config config/DeepSpine_230703_Paran1_leftknee_WBFclip_unified.json
# python train_230703.py --config config/DeepSpine_230703_Paran2_leftknee_WBFclip_unified.json

# OMP_NUM_THREADS=4 python train.py --config config/DeepSpine_230630_Paran1_leftknee_WBFclip_unified.json
# python train.py --config config/DeepSpine_230630_Paran1_leftknee_WBFclip_unified.json
# python train.py --config config/DeepSpine_230630_Paran2_leftknee_WBFclip_unified.json

# python train.py --config config/DeepSpine_230627_Paran1_leftknee_WBFclip_unified.json
# python train.py --config config/DeepSpine_230627_Paran2_leftknee_WBFclip_unified.json

# python train_synthetic.py --config config/DeepSpine_230614_synthetic_paran2_WBFclip_simpleAct.json
# python train_synthetic.py --config config/DeepSpine_230614_synthetic_paran1_WBFclip_simpleAct.json
