ulimit -n 4096
set -x

swapon --show
free -h
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8

EXP_DIR=exps/vcoco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port 29167 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-2branch-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path dataset/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --batch_size 4 \
        --num_workers 2 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --gamma_neg 4 \
        --gamma_pos 0 \
        --with_logic \


