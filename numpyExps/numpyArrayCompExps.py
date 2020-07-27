import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1,10,size=5)
print(compute_reciprocals(values))
big_array = np.random.randint(1,100,size=1000000)
# %timeit compute_reciproals(big_array)


#introducing Ufunctions
###vectorized operation part of numpy###
#%timeit (1/big_array)

#vectorized operations are implemented via the set of ufuncs in numpy

print(np.arange(5) / np.arange(1,6))
#not limited to only 1d arrays
x = np.arange(9).reshape((3,3))
print( 2 ** x)

###exploring Numpy's UFuncs###

#array Arithmetic
x = np.arange(4)
print("x =", x)
print("x + 5=", x+5)
print("x - 5=", x-5)
print("x * 2=", x*2)
print("x / 2=", x/2)
print("x // 2=", x//2) #floor division

print("-x=", -x)
print("x ** 2=", x ** 2)
print("x % 2=", x%2)

#these are rappers for np. functions such as np.add
print("np.add(x, 2)=",np.add(x, 2))

#absolute value
x = np.array([-2,-1,0,1,2])
print(x)
print("abs x",abs(x))
#coresponding NumPy ufunc is np.absolute/ np.abs
print("np abs x", np.abs(x))

#can also handle complex data, returning the magnitude
x = np.array([3-4j, 4-3j, 2+0j, 0 +1j])
print(np.abs(x))

##Trigonometric functions

theta = np.linspace(0, np.pi, 3)

print("theta      =", theta)
print("sin(theta) =", np.sin(theta))
print("cos(theta) =", np.cos(theta))
print("tan(theta) =", np.tan(theta))

theta = [-1, 0, 1]
print("arcsin(theta) =", np.arcsin(theta))
print("arccos(theta) =", np.arccos(theta))
print("arctan(theta) =", np.arctan(theta))


##exponets and logs

x = [1,2,3]
print("x    =",x)
print("e^x  =", np.exp(x))
print("2^x  =", np.exp2(x))
print("3^x  =", np.power(3,x))

x = [1,2,4,10]

print("x=", x)
print("lnx=" ,np.log(x))
print("log2(x)=", np.log2(x))
print("log10(x)=", np.log10(x))

#special cases for percision

x=[0,0.001,0.01, 0.1]
print(x)
print("exp(x)-1", np.expm1(x))
print("log(1+x)", np.log1p(x))

##specialized ufuncs
from scipy import special
#gama functions (generalized factorials) and related
x= [1,5,10]
print("gamma(x)", special.gamma(x))
print("ln|gamma(x)", special.gammaln(x))
print("beta(x,2)", special.beta(x,2))

#error function (intergral of Gaussian)
#it's complement and its inverse
x= np.array([0,0.3,0.7,1.0])
print("erf(x)", special.erf(x))
print("erfc(x)", special.erfc(x))
print("erfinv(x)", special.erfinv(x))

##advance ufunc features
#specifying output
print("advance features")

x= np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2,x,out=y[::2])
print(y)
# for large arrays this is a significant savings vs y[::2] = 2 ** x

print("Aggregates")
x= np.arange(1,6)
print(np.add.reduce(x))
print(np.multiply.reduce(x))
print(np.add.accumulate(x))
print(np.multiply.accumulate(x))

print("outer products")
x = np.arange(1,6)
print(np.multiply.outer(x,x))
