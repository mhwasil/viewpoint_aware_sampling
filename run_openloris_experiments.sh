###########################################################
### Run OpenLORIS-Object sequential dataset experiment 
### on GDumb, Reservoir, Prototype
### Rainbow-Memory (RM), RWalk, VAS, and Naive or Finetune
###########################################################

MEM_SIZES=(256 500 2000 5000)
SEEDS=(43 44 45 46 47)

## GDumb
RESULT_DIR="results/openloris_sequential_gdumb"
for SEED in "${SEEDS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    nohup python train.py --mode=gdumb --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=gdumb --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR &
  done
done

## Finetune or naive
RESULT_DIR="results/openloris_sequential_naive"
for SEED in "${SEEDS[@]}"; do
  python train.py --mode=naive --stream_env online --memory_size=0 --uncertainty_measure="" --mem_manage="" --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx=0 --result_dir=$RESULT_DIR

done

## Reservoir
RESULT_DIR="results/openloris_sequential_reservoir"
for MEM_MANAGE in "${MEM_MANAGES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for MSIZE in "${MEM_SIZES[@]}"; do
      python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=reservoir --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx=0 --result_dir=$RESULT_DIR
    
    done
  done
done

## Prototype
RESULT_DIR="results/openloris_sequential_reservoir"
for MEM_MANAGE in "${MEM_MANAGES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for MSIZE in "${MEM_SIZES[@]}"; do
      python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=prototype --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx=0 --result_dir=$RESULT_DIR
    
    done
  done
done

## Rainbow-Memory (RM)
RESULT_DIR="results/openloris_sequential_rm"

for SEED in "${SEEDS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure=logit_uncertainty --mem_manage=uncertainty --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR
  done
done

## RWalk
RESULT_DIR="results/openloris_sequential_rwalk"

for SEED in "${SEEDS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=rwalk --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=reservoir --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx 3 --result_dir=$RESULT_DIR
  done
done

## VAS
RESULT_DIR="results/openloris_sequential_vas"

for SEED in "${SEEDS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=viewpoint --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=task_balanced --vas_balanced_view --rnd_seed $SEED --dataset=openloris_sequential --exp_name=openloris_sequential --pretrain --cuda_idx=$gpu_counter --result_dir=$RESULT_DIR
     
  done
done

