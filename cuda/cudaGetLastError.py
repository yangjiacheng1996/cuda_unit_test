# -*- coding: utf-8 -*-

import argparse
import ctypes

def test_cudaGetLastError(cuda_lib_pathk):
    # 加载 CUDA 动态库
    libcudart = ctypes.cdll.LoadLibrary(cuda_lib_path)

    # 定义 cudaGetLastError 函数的参数和返回类型
    libcudart.cudaGetLastError.argtypes = []
    libcudart.cudaGetLastError.restype = ctypes.c_int

    # 调用 cudaGetLastError 函数
    error_code = libcudart.cudaGetLastError()

    if error_code == 0:
        print('cudaGetLastError passed.')
    else:
        print(f'cudaGetLastError failed with error code: {error_code}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libpath")
    cmd_arg = parser.parse_args()
    cuda_lib_path = cmd_arg.libpath
    # 用户可控的动态库路径
    #cuda_lib_path = '/usr/local/cuda-11.4/lib64/libcudart.so'


    # 执行测试
    test_cudaGetLastError(cuda_lib_path)
