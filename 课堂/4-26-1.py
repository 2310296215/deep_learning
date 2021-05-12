from sympy import *
from fractions import Fraction

a = Symbol('a')
func = '(2*a-3)**2+(4*a-5)**2+(6*a-8)**2'
res = simplify(eval(func))
learn_rate = 1 / 224
diff_a = diff(res, a)
print(diff_a)
a = 15
while a > 0:
    diff_ = diff_a.evalf(subs={'a': a})
    print('a为：{%.2f}' % a, Fraction(str('%.2f' % a)))
    a = a - learn_rate * diff_
