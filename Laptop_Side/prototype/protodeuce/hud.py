import pygame
import math
import time

import random  # Import the random module

class HUD:
    def __init__(self):
        # Initialize Pygame and GPS serial
        pygame.init()
        
        # Screen dimensions and colors
        self.WIDTH, self.HEIGHT = 800, 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.CYAN = (0, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        # Screen setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("HUD with Speedometer, Tachometer, and Map")

        # Clock for 20 Hz display update
        self.clock = pygame.time.Clock()

        # Load and resize the static map image
        original_map_image = pygame.image.load("Laptop_Side/prototype/protodeuce/hudmap.png")
        self.map_image = pygame.transform.scale(original_map_image, (200, 200))
        self.map_rect = self.map_image.get_rect(topleft=(self.WIDTH - 220, 20))

        # Map bounding coordinates
        self.map_lat_north = 33.58990
        self.map_lat_south = 33.58290
        self.map_lon_west = -101.87942
        self.map_lon_east = -101.87060

        # Speedometer and tachometer settings
        self.speedometer_center = (120, self.HEIGHT - 30)
        self.speedometer_radius = 120
        self.max_speed_mph = 70
        self.display_speed = 0  # Smoothed speed
        self.target_speed = 0

        self.tachometer_value = 1.0
        self.tachometer_max = 2.0
        self.tachometer_width = 180
        self.tachometer_height = 10
        self.tachometer_position = (30, 580)

        # GPS data variables
        self.latitude, self.longitude = 33.5840, -101.8791  # Initial position
        self.prev_latitude, self.prev_longitude = self.latitude, self.longitude
        self.prev_time = time.time()

    def gps_to_pixel(self, lat, lon):
        x_ratio = (lon - self.map_lon_west) / (self.map_lon_east - self.map_lon_west)
        y_ratio = (self.map_lat_north - lat) / (self.map_lat_north - self.map_lat_south)
        x_pixel = self.map_rect.left + int(x_ratio * self.map_rect.width)
        y_pixel = self.map_rect.top + int(y_ratio * self.map_rect.height)
        return x_pixel, y_pixel

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371e3  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c  # Distance in meters

    def draw_speedometer(self, surface, center, radius, speed, max_speed):
        for i in range(0, max_speed + 1, 5):
            angle = math.radians(200 + (340 - 200) * (i / max_speed))
            if i % 10 == 0:
                tick_length = 20
                tick_width = 3
            else:
                tick_length = 10
                tick_width = 1

            start_x = center[0] + int((radius - tick_length) * math.cos(angle))
            start_y = center[1] + int((radius - tick_length) * math.sin(angle))
            end_x = center[0] + int(radius * math.cos(angle))
            end_y = center[1] + int(radius * math.sin(angle))
            pygame.draw.line(surface, self.CYAN, (start_x, start_y), (end_x, end_y), tick_width)

            if i % 10 == 0:
                # Major tick labels
                label_font = pygame.font.SysFont("Arial", 18)
                label = label_font.render(str(i), True, self.CYAN)
                label_x = center[0] + int((radius - 40) * math.cos(angle))
                label_y = center[1] + int((radius - 40) * math.sin(angle))
                surface.blit(label, (label_x - label.get_width() // 2, label_y - label.get_height() // 2))

        # Draw the needle
        needle_angle = math.radians(200 + (340 - 200) * (speed / max_speed))
        needle_x = center[0] + int((radius - 30) * math.cos(needle_angle))
        needle_y = center[1] + int((radius - 30) * math.sin(needle_angle))
        pygame.draw.line(surface, self.RED, center, (needle_x, needle_y), 4)

    def draw_tachometer(self, surface, value, max_value, position, width, height):
        normalized_value = min(max(value / max_value, 0), 1)
        if normalized_value < 0.5:
            color = (
                int(2 * normalized_value * 255),
                255,
                int((1 - 2 * normalized_value) * 255)
            )
        else:
            color = (
                255,
                int((1 - (normalized_value - 0.5) * 2) * 255),
                0
            )
        filled_width = int(normalized_value * width)
        pygame.draw.rect(surface, color, (*position, filled_width, height))
        pygame.draw.rect(surface, self.WHITE, (*position, width, height), 2)

    def run(self):
        running = True
        while running:
            # Placeholder for GPS data retrieval
            gps_lat, gps_lon = None, None  # Replace with actual GPS data retrieval
            if gps_lat is not None and gps_lon is not None:
                self.latitude, self.longitude = gps_lat, gps_lon
            marker_x, marker_y = self.gps_to_pixel(self.latitude, self.longitude)

            # Simulate throttle value
            self.tachometer_value = max(0, min(self.tachometer_max, self.tachometer_value + random.uniform(-0.05, 0.05)))

            # Update speed based on GPS movement
            current_time = time.time()
            distance = self.haversine(self.prev_latitude, self.prev_longitude, self.latitude, self.longitude)
            elapsed_time = current_time - self.prev_time
            if elapsed_time > 1:
                self.target_speed = (distance / elapsed_time) * 2.237  # Convert m/s to mph
                self.prev_latitude, self.prev_longitude, self.prev_time = self.latitude, self.longitude, current_time

            self.display_speed = self.target_speed  # Update display speed

            # Draw HUD elements
            self.screen.fill(self.BLACK)
            self.draw_speedometer(self.screen, self.speedometer_center, self.speedometer_radius, self.display_speed, self.max_speed_mph)
            self.draw_tachometer(self.screen, self.tachometer_value, self.tachometer_max, self.tachometer_position, self.tachometer_width, self.tachometer_height)
            self.screen.blit(self.map_image, self.map_rect)
            pygame.draw.circle(self.screen, self.RED, (marker_x, marker_y), 5)

            # Display speed value in mph and GPS coordinates
            font = pygame.font.SysFont("Arial", 18)
            speed_text = font.render(f"Speed: {self.display_speed:.2f} mph", True, self.WHITE)
            lat_text = font.render(f"Lat: {self.latitude:.5f}", True, self.WHITE)
            lon_text = font.render(f"Lon: {self.longitude:.5f}", True, self.WHITE)
            self.screen.blit(speed_text, (self.WIDTH - 180, 280))
            self.screen.blit(lat_text, (self.WIDTH - 180, 320))
            self.screen.blit(lon_text, (self.WIDTH - 180, 350))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
            self.clock.tick(20)  # Update display at 20 Hz

        pygame.quit()


