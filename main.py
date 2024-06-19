#请在...补充代码
import random

def genpwd(length):
    list = []
    for i in range(length):
        list.append(str(random.randint(1,9)))
    return ''.join(list)

length = eval(input())
random.seed(17)
for i in range(3):
    print(genpwd(length))
