
device=0
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

## Phase 2: Higher Resolution 336 on MVTec AD
base_dir=${depth[0]}_${n_ctx[0]}_${t_n_ctx[0]}_resolution336
save_dir=./checkpoints/${base_dir}/
CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec \
    --data_path ./data/mvtec \
    --save_path ./results/${base_dir}/zero_shot \
    --checkpoint_path ${save_dir}epoch_15.pth \
    --features_list 12 --image_size 336 --depth ${depth[0]} --n_ctx ${n_ctx[0]} --t_n_ctx ${t_n_ctx[0]}
