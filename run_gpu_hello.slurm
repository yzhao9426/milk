#!/bin/bash
#SBATCH --job-name=gpu_hello
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=gpu_hello_%j.out
#SBATCH --error=gpu_hello_%j.err

# 乾淨環境
module purge

# 和剛剛在 login shell 做的一樣：
module load shared
module load cuda12.6/toolkit/12.6.2

# 如果 gpu-env 是用 python39 建的，可以再加這行
# 沒有也可以先試試，不行再加
# module load python39

# 啟動自己的 GPU 虛擬環境
source ~/opt-tests/gpu-env/bin/activate

# 告訴 Numba：NVVM 在哪裡
export NUMBA_CUDA_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so

#（通常不用，但保險：如有需要可再設 driver）
# export NUMBA_CUDA_DRIVER=$CUDA_HOME/lib64/stubs/libcuda.so

cd ~/opt-tests

echo "Running gpu_hello on GPU..."
python gpu_hello.py
