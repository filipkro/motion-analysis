import numpy as np
import matplotlib.pyplot as plt

relu = lambda x: np.max((0,x))
lrelu = lambda x: np.max((0.1*x, x))
sigmoid = lambda x: 1/(1 + np.exp(-x))

r_input = [x for x in range(-7, 8)]
l_input = [x for x in range(-7, 8)]
s_input = [x for x in range(-7, 8)]

relu_out = [relu(x) for x in r_input]
lrelu_out = [lrelu(x) for x in l_input]
zeros = [0 for x in l_input]
sigmoid_out = [sigmoid(x) for x in s_input]

plt.rcParams['xtick.labelsize'] = 'xx-large'
plt.rcParams['ytick.labelsize'] = 'xx-large'
plt.plot(r_input,relu_out, linewidth=3)
plt.figure()
plt.plot(l_input,lrelu_out,linewidth=3)
plt.plot(l_input, zeros,linewidth=2, linestyle='--')
plt.figure()
plt.plot(s_input, sigmoid_out, linewidth=3)

plt.show()
