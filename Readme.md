SPASSC: Semantic Scene Completion (SSC) has emerged as a pivotal task in 3D perception, garnering widespread attention due to its extensive applications, particularly in autonomous driving. Conventional camera-based methodologies often neglect the potential of binocular image feature fusion and the detrimental impact of imprecise voxel features. To address these issues, we introduce a novel approach, Stereo Position Adaptive Semantic Scene Completion (SPASSC), which leverages binocular image pairs to enrich the representation of spatial and semantic characteristics. SPASSC incorporates a Height Adaptive Encoder combined with From-Near-To-Far Scanning. Voxels at higher spatial positions are processed using a Simple Backbone to mitigate interference from empty voxels, while remaining voxels are enhanced through From-Near-To-Far Scanning, a masked self-attention mechanism applied along the frontal view direction. This combination collectively improves feature representation, reduces computational overhead, and enhances reconstruction quality. In addition, a sparse auxiliary head and a semantic segmentation head are used to accelerate semantic learning. Experimental results demonstrate that SPASSC achieves state-of-the-art performance, with IoU of 44.40 and 47.56, and mIoU of 17.24 and 20.16 on the SemanticKITTI and SSCBench-KITTI-360 benchmarks, respectively.

## Acknowledgement

Many thanks to these exceptional open source projects:

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet)
- [Symphonize](https://github.com/hustvl/Symphonies.git)
- [DFA3D](https://github.com/IDEA-Research/3D-deformable-attention.git)
- [VoxFormer](https://github.com/NVlabs/VoxFormer.git)
- [OccFormer](https://github.com/zhangyp15/OccFormer.git)
- [CGFormer](https://github.com/pkqbajng/CGFormer.git)
