import math

def distance(x,y,z):
    return math.sqrt(x*x+y*y+z*z)

if __name__ == '__main__':
    x,y,z = input().split(",")
    d = distance(float(x),float(y),float(z))
    print("{:.2f}".format(d))