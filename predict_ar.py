import numpy as np

PCF = 0.2
LEN_PCF = 10
#SAMPLE_LENGTH = 50

def acf(y):
    y = np.array(y)
    m = np.mean(y)
    var = np.var(y, ddof = 1)
    r = []
    n = len(y)
    for s in range(n-1):
        r.append(np.sum( (y[s+1:] - m) * (y[:n-s-1] - m) ))
    r = np.array(r)
    return r*(1/( (n-1)* (var)))


#print(acf(x)) #[ 0.25 -0.3  -0.45]
def pacf(y):
    r = acf(y)
    r = np.append(r, 0.0)
    y = np.array(y)
    n = len(y)
    f = np.zeros(shape = (n, n), dtype=float)
    f[0,0] = r[0]
    for k in range(1,n):
        sum1 = sum2 = 0
        for j in range(k):
            if k-1 != j :
                f[k-1, j] = f[k-2, j] - f[k-1, k-1] * f[k-2, k-2-j]
            sum1 +=  f[k-1, j]*r[k-j-1]
        f[k, k] = (r[k] - sum1)/(1- np.sum(f[k-1, :k] * r[:k]))
    #print(f)
    pacf = np.array([f[i,i] for i in range(n)])
    return pacf[:-1]
#x = [1,2,3,4]
#print(pacf(x)) #[0.25, -0.38666666666666671, -0.31270903010033446]


def calc_a(endog, order):
    n = len(endog)
    a = np.zeros(shape = (n - 1, order+1), dtype = float)
    a[:,0]=1
    b = endog[1:]
    for j in range(1,order+1):
        for i in range(n-1):
            if i-j >= 0:
                a[i,j] = endog[i-j+1]
    x = np.linalg.lstsq(a,b)[0] #our a: y(n) = a1*y(n-1)+a2*y(n-2)
    return x


def ar(endog, forecast):
    n = len(endog)
    endog = np.array(endog)
    if np.var(endog, ddof = 1) ==0:
        return np.mean(endog)*np.ones(forecast)
    pacf_endog = pacf(endog)
    #print(pacf_endog)
    try:
        order = np.where(abs(pacf_endog)>PCF)[0][-1]+1
    except:
        order = 1
    a = calc_a(endog, order)
    #print(a)
    for i in range(forecast):
        endog = np.append(endog, np.dot(a[1:],endog[:-order-1:-1])+a[0])

    return endog[n:]

#endog = np.array([1,4,9,16,25]) # ok
#endog = np.array([1,1.41,3**0.5,2,5**0.5,6**0.5])
# step = 7
# endog_forecast = arma(endog, step)
#
#
# def compare_vals(name, real, predicted, reconstructed=None):
#         fig = plt.figure()
#         axes = plt.axes()
#         r = np.arange(len(real))
#         axes.set_title(name)
#         axes.set_xlim(0, len(real))
#         axes.grid()
#         axes.plot(r, predicted, label='predicted')
#         if reconstructed != None:
#             axes.plot(r, reconstructed, label='reconstructed')
#         axes.plot(r, real, label='real')
#         axes.legend(loc='upper right', fontsize=16)
#         fig.show()
#
#
# #compare_vals('some', endog, np.append(endog, endog_forecast))
# fig = plt.figure()
# axes = plt.axes()
# data = np.append(endog, endog_forecast)
# r = np.arange(len(data))
# axes.set_xlim(0, len(data))
# axes.grid()
# axes.plot(r,data)
# plt.show()

