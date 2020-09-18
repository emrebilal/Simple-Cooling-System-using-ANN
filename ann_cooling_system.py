import numpy as np

import sys
import Adafruit_DHT
import RPi.GPIO as GPIO

#Raspberry Pi Connections
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        #print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  #(-30 and -50)
  [-10, -40],  # Temp:20 C, Humidity: %10
  [-10, -30],  # Temp:20 C, Humidity: %20
  [-10, -20],  # Temp:20 C, Humidity: %30
  [-10, -10],  # Temp:20 C, Humidity: %40
  [-10, 0],    # Temp:20 C, Humidity: %50
  [-10, 10],   # Temp:20 C, Humidity: %60
  [-10, 20],   # Temp:20 C, Humidity: %70
  [-10, 30],   # Temp:20 C, Humidity: %80
  [-10, 40],   # Temp:20 C, Humidity: %90
  [-5, -40],   # Temp:25 C, Humidity: %10
  [-5, -30],   # Temp:25 C, Humidity: %20
  [-5, -20],   # Temp:25 C, Humidity: %30
  [-5, -10],   # Temp:25 C, Humidity: %40
  [-5, 0],     # Temp:25 C, Humidity: %50
  [-5, 10],    # Temp:25 C, Humidity: %60
  [-5, 20],    # Temp:25 C, Humidity: %70
  [-5, 30],    # Temp:25 C, Humidity: %80
  [-5, 40],    # Temp:25 C, Humidity: %90
  [0, -40],    # Temp:30 C, Humidity: %10
  [0, -30],    # Temp:30 C, Humidity: %20
  [0, -20],    # Temp:30 C, Humidity: %30
  [0, -10],    # Temp:30 C, Humidity: %40
  [0, 0],      # Temp:30 C, Humidity: %50
  [0, 10],     # Temp:30 C, Humidity: %60
  [0, 20],     # Temp:30 C, Humidity: %70
  [0, 30],     # Temp:30 C, Humidity: %80
  [0, 40],     # Temp:30 C, Humidity: %90
  [5, -40],    # Temp:35 C, Humidity: %10
  [5, -30],    # Temp:35 C, Humidity: %20
  [5, -20],    # Temp:35 C, Humidity: %30
  [5, -10],    # Temp:35 C, Humidity: %40
  [5, 0],      # Temp:35 C, Humidity: %50
  [5, 10],     # Temp:35 C, Humidity: %60
  [5, 20],     # Temp:35 C, Humidity: %70
  [5, 30],     # Temp:35 C, Humidity: %80
  [5, 40],     # Temp:35 C, Humidity: %90  
])

all_y_trues = np.array([
  0, #Fan Off
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1, #Fan On
  0,
  0,
  0,
  0,
  1,
  1,
  1,
  1,
  1,
  0,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)

while True:
    humidity, temperature=Adafruit_DHT.read_retry(11,4)
    print ("Temp: {0:0.1f} C Humidity: {1:0.1f} %".format(temperature, humidity))
    
    if temperature >= 20:
      p_temp = temperature - 30
      p_hum = humidity - 50
      value = np.array([p_temp, p_hum])
      fan_status = network.feedforward(value)
      print("Fan Status: %.3f" % fan_status)
    
      if(fan_status > 0.499):
        GPIO.output(17,GPIO.HIGH)
        print("Fan ON!")
      else:
        GPIO.output(17,GPIO.LOW)
        print("Fan OFF!")
    else:
      print("The weather is below 20 degrees!")

