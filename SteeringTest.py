import pygame
# Initialize Pygame
pygame.init()

# Initialize joystick
pygame.joystick.init()

# Check for connected joystick
if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
else:
    # Get the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    # Print the joystick's name and number of axes
    print("Connected to:", joystick.get_name())
    print("Number of axes:", joystick.get_numaxes())

    # Main loop to capture inputs
    running = True
    while running:
        # Handle events (necessary to capture joystick input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # Capture and display values for each axis
        for i in range(joystick.get_numaxes()):
            axis_value = joystick.get_axis(i)
            print(f"Axis {i} value: {axis_value}")
        pygame.time.wait(1000)

pygame.quit()

