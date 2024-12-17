#include <Wire.h>
#include <Adafruit_PWMServoDriver.h> 
#include <stdlib.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(8, 2, 4, 5, 6, 7);

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  120 // Minimum pulse length count (out of 4096)
#define SERVOMAX  520 // Maximum pulse length count (out of 4096)

const int joint1 = 11;
const int joint2 = 12;
const int joint3 = 13;
const int joint4 = 14;
const int joint5 = 15;
const int joint6 = 8;

double joint_1_val = 0.0;
double joint_2_val = 0.0;
double joint_3_val = 0.0;
double joint_4_val = 0.0;
double joint_5_val = 0.0;


unsigned long currentMillis;
long previousMillis = 0;    // set up timers
long interval = 20; 




void setup() {
  Serial.begin(9600);
  Wire.begin();
  //Serial.println("16 channel Servo test!");

  pwm.begin();
  pwm.setPWMFreq(50);  // Analog servos run at ~60 Hz updates

  // Set initial positions for servos on channels 11 to 15
  /*
  pwm.setPWM(11, 0, angleToPulse(10));
  pwm.setPWM(12, 0, angleToPulse(65));
  pwm.setPWM(13, 0, angleToPulse(120));
  pwm.setPWM(14, 0, angleToPulse(100));
  pwm.setPWM(15, 0, angleToPulse(30));
*/
home_position();


}

void loop() {

  /*
  delay(2000);

  // Sweep servo on channel 11 from 10 to 150 degrees
  for (int angle = 10; angle <= 150; angle += 2) {
    delay(20);
    pwm.setPWM(11, 0, angleToPulse(angle));
  }
  delay(1000);

  // Sweep servo on channel 12 from 65 to 90 degrees
  for (int angle = 65; angle <= 90; angle += 1) {
    delay(70);
    pwm.setPWM(12, 0, angleToPulse(angle));
  }
  delay(1000);

  // Sweep servo on channel 13 from 120 to 90 degrees
  for (int angle = 120; angle >= 90; angle -= 2) {
    delay(50);
    pwm.setPWM(13, 0, angleToPulse(angle));
  }
  delay(1000);

  // Sweep servo on channel 14 from 100 to 150 degrees
  for (int angle = 100; angle <= 150; angle += 1) {
    delay(20);
    pwm.setPWM(14, 0, angleToPulse(angle));
  }
  delay(1000);

  // Sweep servo on channel 15 from 30 to 80 degrees
  for (int angle = 30; angle <= 80; angle += 2) {
    delay(40);
    pwm.setPWM(15, 0, angleToPulse(angle));
  }
  delay(1000);
  */
    if (Serial.available() > 0) {
    // Read the incoming angle values from Serial
    String receivedCommand = Serial.readStringUntil('\n');
    if(receivedCommand=="GRAB") grip_close(joint6);
    else if(receivedCommand=="OPEN") grip_open(joint6);
    else{
    float receivedValues[5];

    // Parse the received values
    if (parseCommand(receivedCommand, receivedValues)) {
      joint_1_val = receivedValues[0];
      joint_1_val = joint_1_map(joint_1_val);

      joint_2_val = receivedValues[1];
      joint_2_val = joint_2_map(joint_2_val);

      joint_3_val = receivedValues[2];
      joint_3_val = joint_3_map(joint_3_val);

      joint_4_val = receivedValues[3];
      joint_4_val = joint_4_map(joint_4_val);

      joint_5_val = receivedValues[4];
      joint_5_val = joint_5_map(joint_5_val);

      lcd.clear();
      //displayValues(receivedValues);
    }
      // Set servos to the received angles
       currentMillis = millis();
  //if (currentMillis - previousMillis >= 20)
   //{
      //setServoAngle(joint_1_servo_pin, joint_1);
      pwm.setPWM(joint1, 0, angleToPulse(joint_1_val));
      delay(50);
      //setServoAngle(joint_2_servo_pin, joint_2);
      pwm.setPWM(joint2, 0, angleToPulse(joint_2_val));
      delay(50);
      //setServoAngle(joint_3_servo_pin, joint_3);
      pwm.setPWM(joint3, 0, angleToPulse(joint_3_val));
      delay(50);
      //setServoAngle(joint_4_servo_pin, joint_4);
      pwm.setPWM(joint4, 0, angleToPulse(joint_4_val));
      delay(50);
      //setServoAngle(joint_5_servo_pin, joint_5);
      pwm.setPWM(joint5, 0, angleToPulse(joint_5_val));
      delay(50);
      previousMillis = currentMillis;
   // }
    }
  }



}

//##############################################################//
double joint_1_map(float pos_radian) //outputs in mapped degree
{
  //moveit_sim    :-90  0  90  //<-------pos_radian
  //physical_arm  :-90  0  90 
  return pos_radian*57.2958;  // 57.2958 = 180/pi
}
double joint_2_map(float pos_radian) //outputs in mapped degree
{  
  //moveit_sim    :-90  0  90 //<-------pos_radian
  //physical_arm  :90  0  -90
  return map_double(pos_radian,-1.5708 ,1.5708 ,90 ,-90);  // 57.2958 = 180/pi
}

double joint_3_map(float pos_radian) //outputs in mapped degree
{  
  //moveit_sim    :1.05rad(60.16)  0 -2.09rad(119.74) //<-------pos_radian
  //physical_arm  :-90  0  90
  return map_double(pos_radian ,1.051 ,-2.091 ,-90 ,+90);  // 57.2958 = 180/pi
}
double joint_4_map(float pos_radian) //outputs in mapped degree
{  
  //moveit_sim    :90  0  -90 //<-------pos_radian
  //physical_arm  :-90  0  90
  return map_double(pos_radian, 1.5708 ,-1.5708 ,-90 ,90);  // 57.2958 = 180/pi
}
double joint_5_map(float pos_radian) //outputs in mapped degree
{  
  //moveit_sim    :-90  0  90  //<-------pos_radian
  //physical_arm  :-90  0  90
  return pos_radian*57.2958;  // 57.2958 = 180/pi
}

double map_double(double input,double inputMin,double inputMax,double outputMin,double outputMax)
{
  double normalizedInput = (input - inputMin) / (inputMax - inputMin);
  return outputMin + normalizedInput * (outputMax - outputMin);
}
//##############################################################//

// Function to parse received Serial command
bool parseCommand(String command, float values[]) {
  int index = 0;
  int start = 0;

  // Split the command string by spaces and convert to float
  for (int i = 0; i < command.length(); i++) {
    if (command.charAt(i) == ' ' || i == command.length() - 1) {
      String valueStr = command.substring(start, i + 1);
      values[index] = valueStr.toFloat();  // Convert to float
      index++;
      start = i + 1;

      // Break if we've filled the array
      if (index >= 5) break;
    }
  }
  return (index == 5);  // Return true if we parsed 5 values
}



//##############################################################//

void home_position()
{ 
  delay(1000);
  //driver.setChannelPWM(joint_3_servo_pin,joint_3_servo.pwmForAngle(0.0));
  pwm.setPWM(joint1, 0, angleToPulse(0.0));
  delay(1000);
  //driver.setChannelPWM(joint_2_servo_pin,joint_2_servo.pwmForAngle(0.0));
  pwm.setPWM(joint2, 0, angleToPulse2(0.0));
  delay(1000);
  //driver.setChannelPWM(joint_1_servo_pin,joint_1_servo.pwmForAngle(0.0));
  pwm.setPWM(joint3, 0, angleToPulse(0.0));
  delay(1000);
  //driver.setChannelPWM(joint_4_servo_pin,joint_4_servo.pwmForAngle(0.0));
  pwm.setPWM(joint4, 0, angleToPulse(0.0));
  delay(1000);
  //driver.setChannelPWM(joint_5_servo_pin,joint_5_servo.pwmForAngle(0.0));
  pwm.setPWM(joint5, 0, angleToPulse(0.0));
  delay(1000); 
}

////////////////////////////////////////////////////////////////////////

// Function to map angle to pulse length
int angleToPulse(double ang) {
  int pulse = map(ang, -90, 90, SERVOMIN, SERVOMAX);
  //Serial.print("Angle: "); Serial.print(ang);
  //Serial.print(" Pulse: "); Serial.println(pulse);
  return pulse;
}

int angleToPulse2(double ang) {
  int pulse = map(ang, -90, 90, 170, 530);
  //Serial.print("Angle: "); Serial.print(ang);
  //Serial.print(" Pulse: "); Serial.println(pulse);
  return pulse;
}

void displayValues(float values[]) {
  //lcd.clear();
  //lcd.setCursor(0, 0);
  //lcd.print(values[i]);
  //lcd.print("Received:");
  for (int i = 0; i <6 ; i++) {
    lcd.setCursor(0, 0);  // Print on the second line
    lcd.print("Rec value : ");
    lcd.print(i);
    lcd.setCursor(0,1);
    lcd.print(values[i]);
    //lcd.setCursor(0, 1);  // Print on the second line
    //lcd.print(typeof(values[i]));
    // delay(1000); // Delay to view each value
    // lcd.clear();
    delay(1000);
  }
  delay(500);
}

void grip_close(int gripper_pin){
  pwm.setPWM(gripper_pin, 0, angleToPulse(90));
}

void grip_open(int gripper_pin){
  pwm.setPWM(gripper_pin, 0, angleToPulse(-90));
}

