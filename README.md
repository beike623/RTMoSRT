# RTMoSR
Speed ​​tests carried out on my machine: Rocky linux, ryzen 7 2700, rtx3060 12gb. iterations_warmup 100 iterations_test 1000 img_size [1,3,720,1080]

val_set: urban100

metrics settings: crop_border = 2 test_y_channel = true

it = 150000

warmut: gamma = 0.5 [60000, 120000]



|Name|ssim|psnr|fps|
|-|-|-|-|
|SuperUltraCompact|0.9078|30.17|74.98 cl*|
|RTMoSR_L|0.9095|30.17|97.36|
|RTMoSR_UL|0.9061|29.91|114.04|

cl - memory_format=torch.channels_last
