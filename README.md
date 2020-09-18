# Simple-Cooling-System-using-Basic-Neural-Network
To use the DHT11 sensor, I installed the Adafruit_CircuitPython_DHT module on my board. Then using the codes in the library, I measured the temperature and humidity instantly. In the same code, I used the RPi.GPIO library for fan control.

I included the temperature and humidity values into the artificial neural network algorithm. After giving the input values, I trained the system according to these values. I set output values to 1 and 0. The system continuously entering the temperature and humidity value on the artificial neural network. If the result is approaching 1, the system decides to turn on the cooling system (fan). If the result is close to 0, the system turns off the cooling system (fan).
