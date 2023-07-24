# Common hyperparameters:
epochs=30  # Number of epochs
dataset_shape=256  # CT image size (squared)
res_dir='/mnt/data_jixie1/clma'  # enter your directory for storing result

# Scripts for training FreeSeed (global skip defined in trainer):
num_views=72
network='freeseed_0.5_1_5'
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch \
--master_port 10021 --nproc_per_node 1 \
main.py --epochs $epochs \
--lr 1e-4 --optimizer 'adam' \
--scheduler 'step' --step_size 10 --step_gamma 0.5 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--num_views $num_views \
--network $network \
--net_dict "{'ratio_ginout':0.5,'global_skip':False}" \
--use_mask True --soft_mask False \
--loss 'l2' --trainer_mode 'train' \
--tensorboard_root $res_dir'/freeseed/tb' \
--tensorboard_dir '['$num_views']'$network \
--checkpoint_root $res_dir'/freeseed/ckpt' \
--checkpoint_dir '['$num_views']'$network \
--dataset_path '/mnt/data_jixie1/clma/aapm_tr5410_te526' \
--batch_size 2 --num_workers 4 --log_interval 200 \
--use_tqdm # --use_wandb --wandb_project 'freeseed' --wandb_root $res_dir'/freeseed/wandb'

# Scripts for training FreeNet (global skip defined in network):
num_views=72
network='freenet'
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch \
--master_port 10020 --nproc_per_node 1 \
main.py --epochs $epochs \
--lr 1e-4 --optimizer 'adam' \
--scheduler 'step' --step_size 10 --step_gamma 0.5 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--num_views $num_views \
--network $network \
--net_dict "{'ratio_ginout':0.5,'global_skip':True}" \
--use_mask True --soft_mask False \
--loss 'l2' --trainer_mode 'train' \
--tensorboard_root $res_dir'/freeseed/tb' \
--tensorboard_dir '['$num_views']'$network \
--checkpoint_root $res_dir'/freeseed/ckpt' \
--checkpoint_dir '['$num_views']'$network \
--dataset_path '/mnt/data_jixie1/clma/aapm_tr5410_te526' \
--batch_size 2 --num_workers 4 --log_interval 200 \
--use_tqdm # --use_wandb --wandb_project 'freeseed' --wandb_root $res_dir'/freeseed/wandb'


# Scripts for training DuDoFreeSeed (global skip defined in network):
num_views=72
network='dudofreenet'
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch \
--master_port 10023 --nproc_per_node 1 \
main.py --epochs $epochs \
--lr 1e-4 --optimizer 'adam' \
--scheduler 'step' --step_size 10 --step_gamma 0.5 \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--num_views $num_views \
--network $network \
--net_dict "{'ratio_ginout':0.5}" \
--use_mask True --soft_mask False \
--loss 'l2' --trainer_mode 'train' \
--tensorboard_root $res_dir'/freeseed/tb' \
--tensorboard_dir '['$num_views']'$network \
--checkpoint_root $res_dir'/freeseed/ckpt' \
--checkpoint_dir '['$num_views']'$network \
--dataset_path '/mnt/data_jixie1/clma/aapm_tr5410_te526' \
--batch_size 2 --num_workers 4 --log_interval 200 --save_epochs 1 \
--use_tqdm #--use_wandb --wandb_project 'freeseed' --wandb_root $res_dir'/freeseed/wandb'
