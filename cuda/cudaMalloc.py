# -*- coding: utf-8 -*-

import argparse
import ctypes

def test_cudaMalloc(cuda_lib_path, size):
    # 加载CUDA动态库
    libcudart = ctypes.cdll.LoadLibrary(cuda_lib_path)

    # 定义cudaMalloc函数的参数和返回类型
    libcudart.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libcudart.cudaMalloc.restype = ctypes.c_int

    # 定义CUDA错误码的字符串映射
    cuda_error_codes = {
        0: 'CUDA_SUCCESS',
        1: 'CUDA_ERROR_INVALID_VALUE',
        2: 'CUDA_ERROR_OUT_OF_MEMORY',
        # 添加其他错误码映射...
    }

    # 调用cudaMalloc函数
    d_ptr = ctypes.c_void_p()
    result = libcudart.cudaMalloc(ctypes.byref(d_ptr), size)

    if result == 0:
        print('cudaMalloc passed.')
        print('Device pointer:', d_ptr)
    else:
        error_str = cuda_error_codes.get(result, 'Unknown CUDA error')
        print(f'cudaMalloc failed with error: {error_str}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libpath")
    cmd_arg = parser.parse_args()
    cuda_lib_path = cmd_arg.libpath
    # 用户可控的动态库路径和分配内存的大小
    #cuda_lib_path = '/usr/local/cuda-11.4/lib64/libcudart.so'
    size = 1024

    # 执行测试
    test_cudaMalloc(cuda_lib_path, size)
