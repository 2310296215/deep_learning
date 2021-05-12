from sympy import *
from fractions import Fraction
a = Symbol('a')
f = []
for i in range(3, 92):
    f.append('({}*a-{})**2'.format(i, i + 1))
fun = '+'.join(i for i in f)
print(fun)
res = eval(fun)
f = simplify(res)
print(res)
print(f)
print(-519048/(-2*255341))
print(Fraction(-519048/(-2*255341)))



