###########################################################
### Run CORe50-NI experiment on GDumb, Reservoir, Prototype
### Rainbow-Memory (RM), RWalk, VAS, and Naive or Finetune
###########################################################

MEM_SIZES=(256 500 2000 5000)
RUNS=(run0 run1 run2 run3 run4 run5 run6 run7 run8 run9)

## GDumb
RESULT_DIR="results/core50_ni_inc_gdumb"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=gdumb --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=gdumb --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR    
  done
done

## Naive or finetune
RESULT_DIR="results/core50_naive"
for RUN in "${RUNS[@]}"; do
  python train.py --mode=naive --stream_env online --memory_size=0 --uncertainty_measure="" --mem_manage="" --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx=0 --result_dir=$RESULT_DIR
done

## Reservoir
RESULT_DIR="results/core50_ni_inc_reservoir"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
   python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=$MEM_MANAGE --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR    
  done
done

## Prototype
RESULT_DIR="results/core50_ni_inc_prototype"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
   python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=$MEM_MANAGE --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR    
  done
done

## Rainbow-Memory (RM)
RESULT_DIR="results/core50_ni_inc_rm"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=rm --stream_env online --memory_size=$MSIZE --uncertainty_measure=logit_uncertainty --mem_manage=uncertainty --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR 
  done
done

## RWalk
RESULT_DIR="results/core50_ni_inc_rwalk"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=rwalk --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=reservoir --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --cuda_idx 0 --result_dir=$RESULT_DIR 
  done
done

## VAS
RESULT_DIR="results/core50_ni_inc_vas"
for RUN in "${RUNS[@]}"; do
  for MSIZE in "${MEM_SIZES[@]}"; do
    python train.py --mode=viewpoint --stream_env online --memory_size=$MSIZE --uncertainty_measure="" --mem_manage=task_balanced --rnd_seed=0 --dataset=core50_ni_inc --exp_name=core50_ni_inc --core50_run=$RUN --pretrain --vas_balanced_view --cuda_idx=0 --result_dir=$RESULT_DIR
    
  done
done
