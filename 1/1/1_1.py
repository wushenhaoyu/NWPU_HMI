def getCircleArea(r):
    return r*r*3.1415926

if __name__ == '__main__':
    n = int(input())

    for i in range(n):
        r = float(input())

        print('{:.2f}'.format(getCircleArea(r)))  # 调用getCircleArea并打印结果

    print('END.')