
lst = []
def list1():
    for item in list(range(22)):
        lst.append(item * 2)
        print(lst.pop())
def list2():
    lst = [item * 2 for item in list(range(22))]
    print(lst)
def list3():
    lst = [num for num in range(22) if num%2 == 1]
    print(lst)

def list4():
    lst  = [num if num % 4 == 1 else -num for num in range(1,22) if num %2 == 1]
    print(lst)
if __name__ == "__main__":
    list4()


