#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

#define PCA9685_ADDR 0x40  // Default I2C address of PCA9685
#define PCA9685_MODE1 0x00
#define PCA9685_PRESCALE 0xFE
#define PCA9685_BASE_CHANNEL 0x06
#define FREQUENCY 50  // 50Hz frequency for servos
#define I2C_DEVICE "/dev/i2c-1"  // I2C bus on Jetson Nano

// Servo PWM range (in microseconds)
const int MIN_PULSE = 1000;  // 1ms pulse
const int MAX_PULSE = 2000;  // 2ms pulse

// Function prototypes
int kbhit(void);
char getKey();
void pca9685_init(int i2c_fd);
void set_pwm(int i2c_fd, int channel, int on, int off);
int map_range(int x, int in_min, int in_max, int out_min, int out_max);

int main() {
    // Open I2C device
    int i2c_fd = open(I2C_DEVICE, O_RDWR);
    if (i2c_fd < 0) {
        perror("Failed to open I2C device");
        return 1;
    }

    // Connect to the PCA9685
    if (ioctl(i2c_fd, I2C_SLAVE, PCA9685_ADDR) < 0) {
        perror("Failed to connect to PCA9685");
        return 1;
    }

    // Initialize PCA9685
    pca9685_init(i2c_fd);

    // Starting positions for throttle and steering
    int throttle_position = 1500;  // Neutral throttle
    int steering_position = 1500;  // Centered steering

    printf("Use WASD to control. Press Q to quit.\n");

    while (1) {
        if (kbhit()) {
            char key = getKey();

            // WASD handling
            if (key == 'w' || key == 'W') {
                throttle_position = throttle_position + 50 > MAX_PULSE ? MAX_PULSE : throttle_position + 50;
                printf("Throttle up: %d\n", throttle_position);
            } else if (key == 's' || key == 'S') {
                throttle_position = throttle_position - 50 < MIN_PULSE ? MIN_PULSE : throttle_position - 50;
                printf("Throttle down: %d\n", throttle_position);
            } else if (key == 'a' || key == 'A') {
                steering_position = steering_position - 50 < MIN_PULSE ? MIN_PULSE : steering_position - 50;
                printf("Steering left: %d\n", steering_position);
            } else if (key == 'd' || key == 'D') {
                steering_position = steering_position + 50 > MAX_PULSE ? MAX_PULSE : steering_position + 50;
                printf("Steering right: %d\n", steering_position);
            } else if (key == 'q' || key == 'Q') {
                printf("Quitting...\n");
                break;
            }

            // Set PWM for throttle and steering
            set_pwm(i2c_fd, 0, 0, map_range(throttle_position, 0, 20000, 0, 4095));  // Channel 0 for throttle
            set_pwm(i2c_fd, 1, 0, map_range(steering_position, 0, 20000, 0, 4095));  // Channel 1 for steering
        }

        usleep(100000);  // Sleep for 100ms
    }

    close(i2c_fd);
    return 0;
}

// Function to make keyboard input non-blocking
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

// Function to read a single key press
char getKey() {
    return getchar();
}

// Initialize the PCA9685
void pca9685_init(int i2c_fd) {
    // Wake PCA9685 from sleep and set to normal mode
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_MODE1, 0x00);
    usleep(5000);

    // Set prescale to adjust frequency
    int prescale = (int)(25000000.0 / (4096 * FREQUENCY) - 1.0);
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_PRESCALE, prescale);
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_MODE1, 0x20);
    usleep(5000);
}

// Set PWM value for a channel
void set_pwm(int i2c_fd, int channel, int on, int off) {
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_BASE_CHANNEL + 4 * channel, on & 0xFF);
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_BASE_CHANNEL + 4 * channel + 1, on >> 8);
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_BASE_CHANNEL + 4 * channel + 2, off & 0xFF);
    i2c_smbus_write_byte_data(i2c_fd, PCA9685_BASE_CHANNEL + 4 * channel + 3, off >> 8);
}

// Map a value from one range to another
int map_range(int x, int in_min, int in_max, int out_min, int out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
