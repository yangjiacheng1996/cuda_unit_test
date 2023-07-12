# -*- coding: utf-8 -*-

import ctypes
import argparse


def test_cudaGetErrorString(cuda_lib_path, cuda_error_code):
    # 加载CUDA动态库
    libcudart = ctypes.cdll.LoadLibrary(cuda_lib_path)

    # 定义cudaGetErrorString函数的参数和返回类型
    libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    libcudart.cudaGetErrorString.restype = ctypes.c_char_p

    # 调用cudaGetErrorString函数
    error_string = libcudart.cudaGetErrorString(cuda_error_code)

    if error_string:
        print(f'cudaGetErrorString passed.')
        print(f'Error string for code {cuda_error_code}: {error_string.decode()}')
    else:
        print(f'cudaGetErrorString failed. Invalid error code: {cuda_error_code}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libpath")
    cmd_arg = parser.parse_args()
    cuda_lib_path = cmd_arg.libpath
    # 用户可控的动态库路径和CUDA错误码
    #cuda_lib_path = '/usr/local/cuda-11.4/lib64/libcudart.so'
    cuda_error_code= 2

    # 执行测试
    test_cudaGetErrorString(cuda_lib_path, cuda_error_code)
