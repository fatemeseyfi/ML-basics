import numpy as np
import random

def step(sum_ws):
  if sum_ws >= 0:
    return 1
  else:
    return 0

def sum_weight(x, w):
  sum=0
  for x,w in zip(x,w):
    sum += x*w
  return sum

def update_w(w, err, x, alpha):
  for i in range(len(w)):
    w[i] = w[i] + alpha * err * x[i]
  return w

def perceptron(x,y,w):
  b = 1
  alpha = 0.2
  epochs = 100

  for _ in range(epochs):
    for i in range(x.shape[0]):
      sum_ws = sum_weight(x[i],w) + b
      y_prime = step(sum_ws)
      err = y[i] - y_prime

      update_w(w, err, x[i], alpha)
      b += alpha * err
  
  return w,b

def predict(x,w,b):
  return step(sum_weight(x,w)+b)

x = np.array([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,1,1,1,1,1,1,1])
w = np.array([])


for _ in range(x.shape[1]):
  temp = random.random()
  w = np.append(w, temp)


x_input = [0,1,1]
new_w, new_b = perceptron(x,y,w)
prediction = predict(x_input, new_w, new_b)
print(prediction)

