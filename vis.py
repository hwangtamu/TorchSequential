import matplotlib.pyplot as plt
import numpy as np


x = [2,4,6,8,10,12,16,32,64,128]
y_gru = [(0.48319999999999996, 5.551115123125783e-17), (0.489, 0.0), (0.6136, 0.0), (0.6462000000000001, 1.1102230246251565e-16), (0.8464, 0.0), (0.8684, 0.0), (0.9113000000000001, 1.1102230246251565e-16), (0.9755, 0.0), (0.9902, 0.0), (0.9906, 0.0)]
y_lstm = [(0.4864, 0.0), (0.7619, 0.0), (0.7621, 0.0), (0.7646, 0.0), (0.844, 0.0), (0.8571, 0.0), (0.8911999999999999, 1.1102230246251565e-16), (0.9663, 0.0), (0.9887, 0.0), (0.9906, 0.0)]
y_lstm_d = [(0.6382000000000001, 1.1102230246251565e-16), (0.6655999999999999, 1.1102230246251565e-16), (0.8150999999999999, 1.1102230246251565e-16), (0.8911, 0.0), (0.9155, 0.0), (0.9471000000000002, 1.1102230246251565e-16), (0.9731, 0.0), (0.9825000000000002, 1.1102230246251565e-16), (0.9892, 0.0), (0.9901, 0.0)]
y_gru_d = [(0.48319999999999996, 5.551115123125783e-17), (0.489, 0.0), (0.50459, 0.0906409559746586), (0.6136, 0.0), (0.7785399999999999, 0.1063643944184331), (0.82149, 0.011261478588533571), (0.8524899999999999, 0.025596697052549574), (0.9704599999999999, 0.0027048844707307017), (0.9899700000000001, 0.00042201895692018553), (0.99049, 0.00029816103031752553)]

plt.plot(x, [y[0] for y in y_lstm_d], label='LSTM dropout=0.5')
plt.plot(x, [y[0] for y in y_lstm], label='LSTM dropout=0')
plt.plot(x, [y[0] for y in y_gru], label='GRU dropout=0.5')
plt.plot(x, [y[0] for y in y_gru_d], label='GRU dropout=0')
plt.xscale('log', basex=2)
plt.xlabel('Number of Cells')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()