# Import Classes
from sendcontrols import Send_Control_Data
from gyroscope import Gyroscope

# Import Threading
import threading

# Initialize components
print("Initializing components...")
send_control = Send_Control_Data()
gyroscope    = Gyroscope()
receiver     = ()

# Start threads for each component
print("Configuring threads...")
control_thread   = threading.Thread(target=send_control.start_server)
gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)
receiver_thread  = threading.Thread(target=receiver.)

# Start and manage threads
print("Starting threads...")
control_thread.start()
gyroscope_thread.start()
receiver_thread.start()
