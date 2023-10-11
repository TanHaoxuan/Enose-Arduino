// libraries
#include <dht.h>
#include <MQUnifiedsensor.h>

/************************Hardware Related Macros************************************/
#define         Board                       ("Arduino UNO")
#define         MQ2_pin                     (A0)  //Analog input 0
#define         MQ4_pin                     (A1)  //Analog input 1
#define         DHT11_PIN                    7
/***********************Software Related Macros************************************/
#define         Voltage_Resolution      (5)
#define         ADC_Bit_Resolution      (10) // For arduino UNO/MEGA/NANO
#define         RatioMQ2CleanAir        (9.83) //RS / R0 = 60 ppm
#define         RatioMQ4CleanAir        (4.4) //RS / R0 = 60 ppm 
/*****************************Globals***********************************************/
// Sensor
dht DHT;
MQUnifiedsensor MQ2(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ2_pin, "MQ-2");
MQUnifiedsensor MQ4(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ4_pin, "MQ-4");

// Constant
#define WARM_UP_TIME 20000 //millisecond

#define NUM_SAMPLES 10
#define SAMPLING_FREQ_HZ 2                         // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS 1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)




void setup(){

  // start serial
  Serial.begin(9600);

  //LIG pin
  pinMode(A1, INPUT); 

  //MQ2
  MQ2.setRegressionMethod(1); //_PPM =  a*ratio^b
  MQ2.setA(3616.1); MQ2.setB(-2.675); // Configure the equation to to calculate LPG concentration
    /*
    MQ2 Exponential regression:
    Gas    | a      | b
    H2     | 987.99 | -2.162
    LPG    | 574.25 | -2.222
    CO     | 36974  | -3.109
    Alcohol| 3616.1 | -2.675
    Propane| 658.71 | -2.168
   */
  MQ2.init(); 

  //MQ4
  MQ4.setRegressionMethod(1);
    /*
      Exponential regression:
    Gas    | a      | b
    LPG    | 3811.9 | -3.113
    CH4    | 1012.7 | -2.786
    CO     | 200000000000000 | -19.05
    Alcohol| 60000000000 | -14.01
    smoke  | 30000000 | -8.308
   */
  MQ4.init(); 

  // Pre heat
  Serial.println("Sensors are warming up...");
	delay(WARM_UP_TIME);
  Serial.println("Sensors finish warming up.");

  
  //Calibration
  Serial.print("Calibrating please wait...");
  float calcR0_MQ2 = 0;
  float calcR0_MQ4 = 0;

  for(int i = 1; i<=10; i ++)
  {
    MQ2.update(); // Update data, the arduino will read the voltage from the analog pin
    calcR0_MQ2 += MQ2.calibrate(RatioMQ2CleanAir);
    MQ4.update(); // Update data, the arduino will read the voltage from the analog pin
    calcR0_MQ4 += MQ4.calibrate(RatioMQ4CleanAir);
    Serial.print(".");
  }
  MQ2.setR0(calcR0_MQ2/10);
  Serial.println("MQ2 calibration done.");
  MQ4.setR0(calcR0_MQ4/10);
  Serial.println("MQ4 calibration done.");
  
  if(isinf(calcR0_MQ2)) {Serial.println("Warning: Conection issue, MQ2 R0 is infinite (Open circuit detected) please check your wiring and supply"); while(1);}
  if(calcR0_MQ2 == 0){Serial.println("Warning: Conection issue found, MQ2 R0 is zero (Analog pin shorts to ground) please check your wiring and supply"); while(1);}
  if(isinf(calcR0_MQ4)) {Serial.println("Warning: Conection issue, MQ4 R0 is infinite (Open circuit detected) please check your wiring and supply"); while(1);}
  if(calcR0_MQ4 == 0){Serial.println("Warning: Conection issue found, MQ4 R0 is zero (Analog pin shorts to ground) please check your wiring and supply"); while(1);}



  Serial.println("All initialized");

}

void loop(){

  unsigned long timestamp;
  float MQ2_value; 
  float MQ4_LPG;
  float MQ4_CH4;
  float MQ4_CO;
  float MQ4_Alcohol;
  float MQ4_Smoke;

  float LIG_value;
  MQ2.init();


  // Print header
  Serial.println("timestamp,temp,humd,MQ2_alcohol,MQ4_LPG,MQ4_CH4,MQ4_CO,MQ4_Alcohol,MQ4_Smoke,LIG");


  for (int i = 0; i < NUM_SAMPLES; i++) {

    timestamp = millis();

    // temp&humi - DHT
    int chk = DHT.read11(DHT11_PIN);

    // alcohol - MQ2
    MQ2.update();
    MQ2_value = MQ2.readSensor();

    // MQ4
    MQ4.update();
    MQ4.setA(3811.9); MQ4.setB(-3.113); 
    MQ4_LPG = MQ4.readSensor(); 
    
    MQ4.setA(1012.7); MQ4.setB(-2.786); 
    MQ4_CH4 = MQ4.readSensor(); 

    MQ4.setA(200000000000000); MQ4.setB(-19.05); 
    MQ4_CO = MQ4.readSensor(); 
    
    MQ4.setA(60000000000); MQ4.setB(-14.01); 
    MQ4_Alcohol = MQ4.readSensor(); 
    
    MQ4.setA(30000000); MQ4.setB(-8.308); 
    MQ4_Smoke = MQ4.readSensor(); 
    
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
    Serial.print(MQ4_LPG);
    Serial.print(",");    
    Serial.print(MQ4_CH4);
    Serial.print(",");    
    Serial.print(MQ4_CO);
    Serial.print(",");    
    Serial.print(MQ4_Alcohol);
    Serial.print(",");    
    Serial.print(MQ4_Smoke);
    Serial.print(",");  
    Serial.print(LIG_value);
    Serial.println();

    // Wait for the next sample
    while (millis() < timestamp + SAMPLING_PERIOD_MS);

  }

  Serial.println(); //empty line between samples


}