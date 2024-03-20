import math


def cal(n):
    n += 1
    sum = 0
    list = [ int('6' * j) for j in range(1,n) ]
    return list

def cal1(n):
    list = [1/(i*2+1) for i in range(0,n)]
    return  list
def grade(n):
    try:
        if n<0 or n>100:
            print("成绩有误")
        elif n % 1 != 0:
            print("成绩有误")
        elif n>=90:
            print('A')
        elif n>=80:
            print('B')
        elif n>=70:
            print('C')
        elif n>=60:
            print('D')
        else:
            print('E')
    except BaseException:
        print('成绩有误')
def cal2(n):
    s = n % 17
    if(s < 1):
        print('没有')
    else:
        print(n - s)
def cal3():
    list = [i for i in range (2000 , 3000) if i % 4 == 0 and i % 100 != 0 ]
    print(list)

if __name__ == '__main__':
    print(cal(6))
    print(cal1(6))
    grade(50)
    cal2(89)
    cal3()
