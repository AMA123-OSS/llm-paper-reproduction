# 最终复现实验自动摘要

本文件由 `runs/final_reproduction_summary.py` 从阶段 7-10 的 CSV 自动生成。

## 阶段 7：10 图小规模复现

| run | success_rate | avg_ssim | avg_psnr | avg_elapsed_s |
| --- | --- | --- | --- | --- |
| stage7_steps4_iter5 | 1.000000 | 0.977174 | 41.6998 | 2.35 |
| stage7_steps20_iter50 | 1.000000 | 0.976744 | 41.5805 | 4.59 |

## 阶段 8：Y/UV 消融

| method | success_rate | avg_ssim | avg_psnr | avg_l2 |
| --- | --- | --- | --- | --- |
| rgb_mi_fgsm | 1.000000 | 0.975990 | 42.4469 | 4.0510 |
| y_only | 1.000000 | 0.991483 | 46.5931 | 2.5893 |
| y_uv_embed | 1.000000 | 0.977176 | 41.6994 | 4.7156 |

## 阶段 9：迁移性

| generator | test_model | relation | success_vs_label | conditional_success |
| --- | --- | --- | --- | --- |
| ensemble_y_uv | densenet161 | ensemble_member | 0.900000 | 0.900000 |
| ensemble_y_uv | googlenet | ensemble_member | 0.500000 | 0.500000 |
| ensemble_y_uv | inception_v3 | ensemble_member | 1.000000 | 1.000000 |
| ensemble_y_uv | resnet50 | held_out_black_box | 0.400000 | 0.250000 |
| single_inception_y_uv | densenet161 | transfer_black_box | 0.000000 | 0.000000 |
| single_inception_y_uv | googlenet | transfer_black_box | 0.000000 | 0.000000 |
| single_inception_y_uv | inception_v3 | white_box | 1.000000 | 1.000000 |
| single_inception_y_uv | resnet50 | transfer_black_box | 0.400000 | 0.250000 |

## 阶段 10：恢复验证

当前未能验证 error-free recovery。缺失前置条件：

T_flag_persisted, arithmetic_dictionary_persisted, compressed_bit_length_persisted, standalone_recovery_entry
