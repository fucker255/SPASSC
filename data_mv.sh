# soft linking datasets and pretrained models
# please make sure the paths in config files are correct
mkdir pretrain
mkdir -p dataset/SSCBenchKITTI360
mkdir -p dataset/semantickitti

ln -s /mnt/vepfs/datasets-v2/tmp/wyh/datasets_ssc/semantickitti/semantickitti/* dataset/semantickitti/
ln -s /mnt/vepfs/datasets-v2/tmp/wyh/datasets_ssc/KITTI360/sscbench-kitti/* dataset/SSCBenchKITTI360/
ln -s /mnt/vepfs/datasets-v2/tmp/wyh/datasets_ssc/*.pth pretrain/