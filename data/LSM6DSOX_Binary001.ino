// Basic demo for accelerometer & gyro readings from Adafruit
// LSM6DSOX sensor
// LSM6DSOX: Vin--5v; gnd--gnd; SCL-A5; SDA-A4
// SD: 3.3V--5V; 13--D13; 12--D12; 11--D11; 8--D10

#include <Adafruit_LSM6DSOX.h>
#include <SPI.h>
#include <SD.h>


File myFile;

struct datastore {
    float  accelx;
    float  accely;
    float  accelz;
    float  gyrox;
    float  gyroy;
    float  gyroz;
};


Adafruit_LSM6DSOX sox;
void setup() {
  
    Serial.begin(9600);
    Serial.print("Initializing SD card...");

  
if (!sox.begin_I2C()) {
    while (1);
  } 
  if (!SD.begin()) {
    while (1);
  }


}
void loop() {
  //  /* Get a new normalized sensor event */
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  sox.getEvent(&accel, &gyro, &temp);

//  myFile = SD.open("test003.dat", FILE_WRITE);
  myFile = SD.open("datalog.dat", FILE_WRITE);

struct datastore myData;
   myData.accelx = accel.acceleration.x;
   myData.accely = accel.acceleration.y;
   myData.accelz = accel.acceleration.z;
   myData.gyrox = gyro.gyro.x;
   myData.gyroy = gyro.gyro.y;
   myData.gyroz = gyro.gyro.z;
   
   myFile.write((const uint8_t * )&myData, sizeof(myData));
   delay(500);
   Serial.println("Data has been written to SD Card");
   myFile.close();
   
//  myFile.print(accel.acceleration.x);
//  myFile.print(",");
//  myFile.print(accel.acceleration.y);
//  myFile.print(",");
//  myFile.print(accel.acceleration.z);
//  myFile.print(",");
//  myFile.print(gyro.gyro.x);
//  myFile.print(",");
//  myFile.print(gyro.gyro.y);
//  myFile.print(",");
//  myFile.print(gyro.gyro.z);
//  myFile.println();
//  myFile.close();

  delay(1);
   
}
