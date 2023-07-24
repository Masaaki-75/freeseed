dataset_shape=256  # CT image size (squared)
res_dir='/mnt/data_jixie1/clma'  # enter your directory for storing result


num_views=72
network='freeseed_0.5_1_5'
CUDA_VISIBLE_DEVICES="2" python -m torch.distributed.launch \
--master_port 18013 --nproc_per_node 1 \
main.py \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' --num_views $num_views \
--tensorboard_root $res_dir'/freeseed/tb' \
--tensorboard_dir '['$num_views']'$network \
--dataset_path '/mnt/data_jixie1/clma/aapm_tr5410_te526' \
--net_checkpath $res_dir'/freeseed/ckpts/[72]freeseed-free.pth' \
--tester_save_path $res_dir'/aapm_vis' \
--tester_save_name '['$num_views']'$network \
--tester_save_image 