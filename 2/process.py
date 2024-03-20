import time
scale = 10
def progress1():
    print("------------boom has been planted!--------------")
    for i in range(scale + 1):
        a = '*' * i
        b = '.' * (scale * i)
        c = (i / scale) * 100
        print("{:^3.0f}%[{}->{}]".format(c,a,b))
        time.sleep(0.1)
    print("------------boom has been confused!-------------")

def progress2():
    for i in range(101):
        print("\r{:3}%".format(i),end="")
        time.sleep(0.1)
if __name__ == '__main__':
    progress2()