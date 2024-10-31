import serial
import keyboard
import time
import sys
import pygame

arduino_port = 'COM3'
baud_rate = 9600

time.sleep(2)

try:
    ser = serial.Serial(arduino_port, baud_rate)
    print("Connected to Arduino on port", arduino_port)
except serial.SerialException as e:
    print(f"Failed to connect to {arduino_port}: {e}")
    exit()
    
pygame.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
else:
    # Get the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Connected to:", joystick.get_name())
    print("Number of axes:", joystick.get_numaxes())
    
    running = True
    axis_value = [0] * joystick.get_numaxes()  
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for i in range(joystick.get_numaxes()):
            axis_value[i] = joystick.get_axis(i)
        
        axis_value_str = ','.join([f"{value:.2f}" for value in axis_value]) 
        ser.write((axis_value_str + '\n').encode())  
        print(f"Sent to Arduino: {axis_value_str}")
        
        pygame.time.wait(100)
        
        if keyboard.is_pressed('q'):
            print("Exiting...")
            sys.stdout.flush() 
            break

ser.close()
pygame.quit()
