# -*- coding: utf-8 -*-

import time

def timer(func):
    def wrapper(cuda_lib_path, test_file):
        start_time = time.time()
        result = func(cuda_lib_path, test_file)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result
    return wrapper

# 示例用法
if __name__ == '__main__':
    @timer
    def my_function():
        # 在这里执行你的函数代码
        time.sleep(2)

    # 调用带有计时装饰器的函数
    my_function()
