TBSFNet
*******************************************************************
requirements:

source /home/datacenter/hdd/data_yurun/miniconda/bin/activate
conda create -n fingerprint python=3.10
conda activate fingerprint
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pillow
conda install numpy
pip install opencv-python
conda install scipy
conda install tqdm
conda install matplotlib
pip install albumentations
pip install opencv-contrib-python
pip install fplab ==0.0.3.2
pip install aircv
pip install torchinfo

*******************************************************************
train:

autodl12282
source /root/miniconda3/bin/activate
conda activate fingerprint
cd /root/autodl-tmp/2025_Proposed/

python data_preparation.py --source_dir /home/datacenter/hdd/data_yurun/train_datasets/anguli_nbis_v4_verbose/ --output_dir /home/datacenter/hdd/data_yurun/augment_results/anguli_nbis_v4_verbose --fixed_seed 2024 --minutiae_sigma 8.0

Proposed_20250222020452
cd /root/autodl-tmp/2025_Proposed/
CUDA_VISIBLE_DEVICES=0 python train.py --input_files_dir /home/datacenter/hdd/data_yurun/augment_results/anguli_nbis_v4_verbose/ --output_dir /root/autodl-tmp/train_results/ --batch_size 8 --gpu_ids 0 --learning_rate 0.0001 --epoch_end 10 --epoch_fixed_lr 5 --prepare_data_freq 20 --test_freq 10 --save_freq 20 --fixed_seed 2024 --name Proposed --epoch_stage1 2 --epoch_stage2 4 --m_lr 1 --m_r 0 --m8_r 1 --r_lr 1 --s_lr 0 --o_l_lr 1 --f_l_lr 0 --o_f2_lr 1 --f_f2_lr 0 --image_size_max 448 --image_pad_size 512 512 --res_num_n 3 6 3 3 6 --no_inverse

*******************************************************************
test:
test.ipynb