# -*- coding: utf-8 -*-

import argparse
import ctypes
import numpy as np


def test_cudaMemcpy(cuda_lib_path, src, dest):
    # 加载CUDA动态库
    libcudart = ctypes.cdll.LoadLibrary(cuda_lib_path)

    # 定义cudaMemcpy函数的参数和返回类型
    libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libcudart.cudaMemcpy.restype = ctypes.c_int

    # 将源数据复制到设备内存
    src_ptr = ctypes.c_void_p(src.ctypes.data)
    dest_ptr = ctypes.c_void_p(dest.ctypes.data)
    size = ctypes.c_size_t(src.nbytes)
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyHostToHost = 3
    cudaMemcpyDeviceToDevice = 4

    # Host to Device
    result = libcudart.cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyHostToDevice)
    if result == 0:
        print('cudaMemcpy (Host to Device) passed.')
    else:
        print('cudaMemcpy (Host to Device) failed.')

    # Device to Host
    result = libcudart.cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyDeviceToHost)
    if result == 0:
        print('cudaMemcpy (Device to Host) passed.')
    else:
        print('cudaMemcpy (Device to Host) failed.')

    # Host to Host
    result = libcudart.cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyHostToHost)
    if result == 0:
        print('cudaMemcpy (Host to Host) passed.')
    else:
        print('cudaMemcpy (Host to Host) failed.')

    # Device to Device
    result = libcudart.cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyDeviceToDevice)
    if result == 0:
        print('cudaMemcpy (Device to Device) passed.')
    else:
        print('cudaMemcpy (Device to Device) failed.')

    # 验证数据是否正确复制
    if np.array_equal(dest, src):
        print('Data verification passed.')
    else:
        print('Data verification failed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libpath")
    cmd_arg = parser.parse_args()
    cuda_lib_path = cmd_arg.libpath
    # 用户可控的动态库路径和数据
    #cuda_lib_path = '/usr/local/cuda-11.4/lib64/libcudart.so'

    src_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dest_data = np.zeros_like(np.array([1, 2, 3, 4, 5], dtype=np.int32), )

    # 执行测试
    test_cudaMemcpy(cuda_lib_path,src_data,dest_data)
