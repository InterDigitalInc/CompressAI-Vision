# this add the proper cuda directories to paths
CUDA_VERSION="11.3"

# cuda version is an optional argument, default version 11.3
# example: source env_cuda.sh "11.8"

if [ $# == 1 ];then
    CUDA_VERSION=$1
fi

export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib:${LD_LIBRARY_PATH}