from sympy import *

a = Symbol('a')
b = Symbol('b')
f = []
for i in range(3, 92):
    f.append('(' + str(i) + '*' + 'a' + '+' + 'b' + '-' + str(i + 1) + ')' + '**2')
fun = '+'.join(i for i in f)
res = eval(fun)
f = simplify(res)
print('原表达式：', res)
print('化简之后：', f)
diff_a = diff(f, a)
diff_b = diff(f, b)
print('求偏导')
print(diff_a, '= 0')
print(diff_b, '= 0')
print(solve([diff_a, diff_b], [a, b]))

