# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from decorator import timer

@timer
def run_test(cuda_lib_path, test_file):
    # 检测 Python 版本
    python_cmd = 'python' if sys.version_info.major == 2 else 'python3'

    # 使用 subprocess 运行测试文件
    process = subprocess.Popen([python_cmd, test_file, cuda_lib_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # 获取测试结果
    result = stdout.decode().strip()

    # 返回测试结果
    return result


def compare_results(result1, result2):
    # 比较两个结果
    if result1 == result2:
        print('Results match.')
        return True
    else:
        print('Results do not match.')
        return False


def main(api_name: str):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(project_dir, "cuda",f"{api_name}.py")
    cuda_lib_path1 = '/path/to/cuda_lib1.so'
    cuda_lib_path2 = '/path/to/cuda_lib2.so'
    # 运行第一个测试文件
    result1 = run_test(cuda_lib_path1, test_file)
    # 运行第二个测试文件
    result2 = run_test(cuda_lib_path2, test_file)
    return compare_results(result1, result2)

if __name__ == "__main__":
    main("cudaMemcpy")


