// libraries
#include <dht.h>
#include <MQUnifiedsensor.h>

/************************Hardware Related Macros************************************/
#define Board ("Arduino UNO")
#define         MQ2_pin                     (A0)  //Analog input 4 of your arduino
#define         DHT11_PIN                    7
/***********************Software Related Macros************************************/
#define         Voltage_Resolution      (5)
#define         ADC_Bit_Resolution      (10) // For arduino UNO/MEGA/NANO
#define         RatioMQ2CleanAir        (9.83) //RS / R0 = 60 ppm
/*****************************Globals***********************************************/
//Declare Sensor
dht DHT;
MQUnifiedsensor MQ2(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ2_pin, "MQ-2");


// constant
#define  NUM_SAMPLES 10
#define SAMPLING_FREQ_HZ 2                         // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS 1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)




void setup(){

  // start serial
  Serial.begin(9600);

  //LIG pin
  pinMode(A1, INPUT); 
  MQ2.setRegressionMethod(1); //_PPM =  a*ratio^b
  MQ2.setA(574.25); MQ2.setB(-2.222); // Configure the equation to to calculate LPG concentration
  MQ2.init(); 

  //
  Serial.println("MQ2 is warming up...");
	delay(20000);
  
  //calibration
  Serial.print("Calibrating please wait.");
  float calcR0 = 0;
  for(int i = 1; i<=10; i ++)
  {
    MQ2.update(); // Update data, the arduino will read the voltage from the analog pin
    calcR0 += MQ2.calibrate(RatioMQ2CleanAir);
    Serial.print(".");
  }
  MQ2.setR0(calcR0/10);
  Serial.println("  done!.");
  
  if(isinf(calcR0)) {Serial.println("Warning: Conection issue, R0 is infinite (Open circuit detected) please check your wiring and supply"); while(1);}
  if(calcR0 == 0){Serial.println("Warning: Conection issue found, R0 is zero (Analog pin shorts to ground) please check your wiring and supply"); while(1);}
  /*****************************  MQ CAlibration ********************************************/ 

  //MQ2.serialDebug(true);


  Serial.println("All initialized");

}

void loop(){

  unsigned long timestamp;
  float MQ2_value; 
  float LIG_value;
  MQ2.init();


  // Print header
  Serial.println("timestamp,temp,humd,MQ2,LIG");


  for (int i = 0; i < NUM_SAMPLES; i++) {

    timestamp = millis();

    // temp&humi - DHT
    int chk = DHT.read11(DHT11_PIN);

    // MQ2
    //MQ2_value = analogRead(MQ2_pin); 
    MQ2.update();
    MQ2_value = MQ2.readSensor();

    // LIG
    LIG_value = analogRead(A1);



    // Print CSV data with timestamp
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(DHT.temperature);
    Serial.print(",");
    Serial.print(DHT.humidity);
    Serial.print(",");
    Serial.print(MQ2_value);
    Serial.print(",");
    Serial.print(LIG_value);
    Serial.println();

    // Wait for the next sample
    while (millis() < timestamp + SAMPLING_PERIOD_MS);

  }

  Serial.println(); //empty line between samples


}