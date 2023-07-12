# -*- coding: utf-8 -*-

import argparse
import ctypes

def test_cudaFree(cuda_lib_path, d_ptr):
    # 加载 CUDA 动态库
    libcudart = ctypes.cdll.LoadLibrary(cuda_lib_path)

    # 定义 cudaFree 函数的参数和返回类型
    libcudart.cudaFree.argtypes = [ctypes.c_void_p]
    libcudart.cudaFree.restype = ctypes.c_int

    # 调用 cudaFree 函数
    result = libcudart.cudaFree(d_ptr)

    if result == 0:
        print('cudaFree passed.')
    else:
        print('cudaFree failed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libpath")
    cmd_arg = parser.parse_args()
    cuda_lib_path = cmd_arg.libpath
    # 用户可控的动态库路径和设备指针
    #cuda_lib_path = '/usr/local/cuda-11.4/lib64/libcudart.so'
    d_ptr= ctypes.c_void_p(123)  # 示例，将 d_ptr 设置为实际的设备指针

    # 执行测试
    test_cudaFree(cuda_lib_path, d_ptr)
