# A modification of HPLFlowNet
My purpose is to make it adaptive to the newer python, pytorch, cuda versions.
* [Original Repo](https://github.com/laoreja/HPLFlowNet)
* [Original Paper](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf)

## Tested Environment
|python|pytorch|cuda|cffi|
|---|---|---|---|
|3.9.18|1.13.0|11.6|1.16.0|

* Installation on Ubuntu:
```bash
pip install torch numba cffi mayavi joblib pypng
```
* Setup:
```bash
cd models; python3 build_khash_cffi.py; cd ..
```
## Data preprocess

* FlyingThings3D:
Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
. They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

* KITTI Scene Flow 2015
Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```


### Trained models
Out trained model can be downloaded in the [`trained_models`](https://github.com/laoreja/HPLFlowNet/tree/master/trained_models) folder.

### Inference
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Set `resume` to be the path of your trained model or our trained model in `trained_models`. Then run
```bash
python3 main.py configs/test_xxx.yaml
```

Current implementation only supports `batch_size=1`.

### Train
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Then run
```bash
python3 main.py configs/train_xxx.yaml
```

### Visualization
If you set `TOTAL_NUM_SAMPLES` in `evaluation_bnn.py` to be larger than 0. Sampled results will be saved in a subdir of your checkpoint directory, `VISU_DIR`.

Run
```bash
python3 visualization.py VISU_DIR
``` 

## Citation

If you use this code for your research, please cite our paper.


```
@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
```
## Acknowledgments
Our permutohedral lattice implementation is based on [Fast High-Dimensional Filtering Using the Permutohedral Lattice](http://graphics.stanford.edu/papers/permutohedral/). The [BilateralNN](https://github.com/MPI-IS/bilateralNN) implementation is also closely related.
Our hash table implementation is from [khash-based hashmap in Numba](https://github.com/synapticarbors/khash_numba).
