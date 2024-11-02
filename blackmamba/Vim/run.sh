#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/zxy/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/zxy/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/zxy/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/zxy/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate Vit-env &&
cd ./vim &&

# eval
# python main.py --eval --resume /home/zxy/.cache/huggingface/hub/models--hustvl--Vim-tiny-midclstok/snapshots/07c00e0e4ea2973d8e343afdd807128a57bc9fd5/vim_t_midclstok_76p1acc.pth \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --device cuda:7 \
#     --data-path /home/zxy/datas/ILSVRC/imagenet-1k/data

# train
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 256 --epochs 10 --drop-path 0.0 --weight-decay 0.1 --num_workers 16 \
--data-path /home/zxy/datas/ILSVRC/imagenet-1k/data \
--output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--no_amp

# finetune
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=8 --use_env main.py \
# --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
# --batch-size 32 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 \
# --num_workers 16 \
# --data-path /home/zxy/datas/ILSVRC/imagenet-1k/data \
# --output_dir ./output/vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
# --epochs 30 \
# --finetune /home/zxy/.cache/huggingface/hub/models--hustvl--Vim-tiny-midclstok/snapshots/07c00e0e4ea2973d8e343afdd807128a57bc9fd5/vim_t_midclstok_76p1acc.pth \
# --no_amp
