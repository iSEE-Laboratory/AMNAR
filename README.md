# Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks

(CVPR 2025) Official implementation of Paper ''Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks''

CVPR version: [Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Modeling_Multiple_Normal_Action_Representations_for_Error_Detection_in_Procedural_CVPR_2025_paper.html)

arXiv version: [Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks](https://arxiv.org/abs/2503.22405)

If you find this paper helpful, please cite our work:

```plaintext
@article{huang2025modeling,
  title={Modeling Multiple Normal Action Representations for Error Detection in Procedural Tasks},
  author={Huang, Wei-Jin and Li, Yuan-Ming and Xia, Zhi-Wei and Tang, Yu-Ming and Lin, Kun-Yu and Hu, Jian-Fang and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2503.22405},
  year={2025}
}
```

## 1 Prepare and Process Data

Download Datasets from official release: [EgoPER](https://github.com/robert80203/EgoPER_official), [HoloAssist](https://holoassist.github.io/), [CaptainCook4D](https://github.com/CaptainCook4D/downloader)


EgoPER Dataset File Structure

```plaintext
EgoPER/
├── coffee
│   ├── features_10fps_dinov2
│   ├── features_10fps_new
│   ├── frames_10fps_new
│   ├── test.txt
│   ├── training.txt
│   ├── trim_start_end.txt
│   ├── trim_videos
│   └── validation.txt
├── oatmeal
│   ├── features_10fps_dinov2
│   ├── features_10fps_new
│   ├── frames_10fps_new
│   ├── test.txt
│   ├── training.txt
│   ├── trim_start_end.txt
│   ├── trim_videos
│   └── validation.txt
├── ......
```


## 2 Training

A training example of `tea` task in EgoPER

- Train Action Segmentation Model (ActionFormer)

```bash
python train.py ./configs/EgoPER/tea_aod_online.yaml --output af200
```

- Train Error Detection Model

```bash
python train.py ./configs/EgoPER/tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200.yaml --resume ./ckpt/EgoPER/tea_aod_online_af200/epoch_205.pth.tar --output 1st 
```

Also, you can choose to download our weights to reproduce our work. [Google Drive](https://drive.google.com/drive/folders/1DrnDhNWq1MDmtFpjwOMz_VuQo5PeAD_a?usp=sharing)

## 3 Inference

An inference example of `tea` task in EgoPER

- Use Action Segmentation Model to get the segmentation result

```bash
python test.py ./configs/EgoPER/tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200.yaml ./ckpt/EgoPER/tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200_1st
```

- Use Error Detection Model to detect errors

```bash
python test_ed.py ./configs/EgoPER/tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200.yaml ./ckpt/EgoPER/tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200_1st
```

- Calculate Test Metrics

```
python metric_vis_multiprocess.py --task tea --dirname tea_aod_rebuild_ca_sigt_online_clusterCenter_it0.6_addedMax_norm_unfreeze_winlen32_dila3_dilaLayer5_fr5_cu10_cus0.40_m0.9_dp0.1_e200_1st/ -as -ed 
```

More examples can be found in `run_sh_EgoPER/run_clusterCenter_it0.6_addedMax_ca_usenorm_unfreeze_winlen32_dilation3_dilalayers5_fr5_cu10_cus0.40_m0.9_dp0.1_e200.sh`

## Note

The implementation of AMNAR's code is based on [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [EgoPED](https://github.com/robert80203/EgoPER_official). We recommend reading the code of these two works to help with understanding.
