import time

def function_test(func):
    def wrapper(*args, **kwargs):
        print(f"Function {func.__name__} is running.")
        time_start=time.perf_counter()
        result=func(*args, **kwargs)
        time_end=time.perf_counter()
        time_value=time_end-time_start
        print(f"The function takes {time_value} seconds.")
        return result
    return wrapper

@function_test
def useless():
    sum=0
    for i in range(1,1000):
        for j in range(1,1000):
            sum+=(i%j)
    print(f"The result is {sum}.")

if __name__ == "__main__":
    useless()





