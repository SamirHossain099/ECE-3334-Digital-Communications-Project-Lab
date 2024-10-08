#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_I2CDevice.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int throttleChannel = 0;  
const int steeringChannel = 1;  

const int MIN_PULSE = 1000;      // Minimum pulse width (1ms, full reverse/left)
const int NEUTRAL_PULSE = 1500;  // Neutral position (1.5ms)
const int MAX_PULSE = 2000;      // Maximum pulse width (2ms, full forward/right)

int throttlePosition = NEUTRAL_PULSE;
int steeringPosition = NEUTRAL_PULSE;

int mapAxisToPulse(float axisValue, int minPulse, int neutralPulse, int maxPulse) {
  int pulseWidth = map(axisValue * 1000, -1000, 1000, minPulse, maxPulse);  // Map axis range to PWM
  if (pulseWidth < minPulse) {
    pulseWidth = minPulse;
  } else if (pulseWidth > maxPulse) {
    pulseWidth = maxPulse;
  }
  return pulseWidth;
}

void updateServos() {
  pwm.setPWM(throttleChannel, 0, pulseWidthToPWM(throttlePosition));
  pwm.setPWM(steeringChannel, 0, pulseWidthToPWM(steeringPosition));
}

int pulseWidthToPWM(int pulseWidth) {
  return (int)((pulseWidth * 4096.0) / 20000.0); 
}

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(46);  // PWM frequency to 46Hz

  updateServos();
}

void loop() {
  if (Serial.available()) {
    String inputString = Serial.readStringUntil('\n');  
    inputString.trim();  
    
    //expecting axis1,axis2,axis3
    int firstCommaIndex = inputString.indexOf(',');
    int secondCommaIndex = inputString.indexOf(',', firstCommaIndex + 1);
    
    if (firstCommaIndex > 0 && secondCommaIndex > firstCommaIndex) {
      String axis1String = inputString.substring(0, firstCommaIndex);
      String axis2String = inputString.substring(firstCommaIndex + 1, secondCommaIndex);
      String axis3String = inputString.substring(secondCommaIndex + 1);

      float axis1Value = axis1String.toFloat();  // Steering axis 1
      float axis2Value = axis2String.toFloat();  // Forward Throttle axis 2
      float axis3Value = axis3String.toFloat();  // Reverse Throttle axis 3

      // Map axis1 (steering) to steering position (continuous mapping)
      steeringPosition = mapAxisToPulse(axis1Value, MIN_PULSE, NEUTRAL_PULSE, MAX_PULSE);
      
      if (axis2Value < 1.0) {  // Forward throttle control
        throttlePosition = mapAxisToPulse(1.0 - axis2Value, NEUTRAL_PULSE, NEUTRAL_PULSE, MAX_PULSE);  // 1.0 to -1.0 (neutral to forward)
      } else if (axis3Value < 1.0) {  // Reverse throttle control
        throttlePosition = mapAxisToPulse(axis3Value, MIN_PULSE, NEUTRAL_PULSE, NEUTRAL_PULSE);  // 1.0 to -1.0 (neutral to reverse)
      } else {
        throttlePosition = NEUTRAL_PULSE;  // Both axes neutral, set throttle to neutral
      }
      updateServos();
    }
  }
}


