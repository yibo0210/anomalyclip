
device=0
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

## Phase 3: Attention Adapter on MVTec AD
base_dir=${depth[0]}_${n_ctx[0]}_${t_n_ctx[0]}_attn_adapter
save_dir=./checkpoints/${base_dir}/
mkdir -p ${save_dir}
CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec \
    --train_data_path ./data/mvtec \
    --save_path ${save_dir} \
    --features_list 12 --image_size 224 --batch_size 16 --print_freq 1 \
    --epoch 15 --save_freq 1 --depth ${depth[0]} --n_ctx ${n_ctx[0]} --t_n_ctx ${t_n_ctx[0]}
