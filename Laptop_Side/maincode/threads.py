# Import Classes
from Laptop_Side.maincode.sendcontrols import Send_Control_Data
from Laptop_Side.maincode.gyroscope import Gyroscope
# Import Threading
import threading

# Initialize components
send_control = Send_Control_Data()
gyroscope = Gyroscope()

# Start threads for each component
control_thread = threading.Thread(target=send_control.start_server)
gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)

# Start and manage threads
control_thread.start()
gyroscope_thread.start()
