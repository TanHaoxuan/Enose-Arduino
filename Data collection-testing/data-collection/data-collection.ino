// libraries
#include <dht.h>
#include <MQUnifiedsensor.h>

/************************Hardware Related Macros************************************/
#define         Board                       ("Arduino UNO")
#define         MQ2_pin                     (A0)  //Analog input 0
#define         MQ4_pin                     (A1)  //Analog input 1
#define         MQ5_pin                     (A2)  //Analog input 1
#define         DHT11_PIN                    7
/***********************Software Related Macros************************************/
#define         Voltage_Resolution      (5)
#define         ADC_Bit_Resolution      (10)   // For arduino UNO/MEGA/NANO
#define         RatioMQ2CleanAir        (9.83) //RS / R0 = 60 ppm
#define         RatioMQ4CleanAir        (4.4)  //RS / R0 = 60 ppm 
#define         RatioMQ5CleanAir        (6.5)  //RS / R0 = 6.5 ppm 
/*****************************Globals***********************************************/
// Sensor
dht DHT;
MQUnifiedsensor MQ2(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ2_pin, "MQ-2");
MQUnifiedsensor MQ4(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ4_pin, "MQ-4");
MQUnifiedsensor MQ5(Board, Voltage_Resolution, ADC_Bit_Resolution, MQ5_pin, "MQ-5");

// Constant
#define WARM_UP_TIME 20000 //millisecond

#define NUM_SAMPLES 10
#define SAMPLING_FREQ_HZ 2                         // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS 1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)

float MQ_setup_calib(MQUnifiedsensor MQ, String name, int regression_method, float RsR0CleanAir);



void setup(){

  // start serial
  Serial.begin(9600);

  //LIG pin
  pinMode(A1, INPUT); 

  // Pre heat
  Serial.println("Sensors are warming up...");
	delay(WARM_UP_TIME);
  Serial.println("Sensors finish warming up.");

  
  //Calibration
  float calcR0_MQ = 0;

  calcR0_MQ = MQ_setup_calib(MQ2, "MQ-2", 1, RatioMQ2CleanAir);
  MQ2.setR0(calcR0_MQ/10);
  calcR0_MQ = MQ_setup_calib(MQ4, "MQ-4",1, RatioMQ4CleanAir);
  MQ4.setR0(calcR0_MQ/10);
  calcR0_MQ = MQ_setup_calib(MQ5, "MQ-5",1, RatioMQ5CleanAir);
  MQ5.setR0(calcR0_MQ/10);

  Serial.println("All initialized");
  Serial.println("timestamp,temp,humd,MQ2_alcohol,MQ2_H2,MQ2_Propane,MQ4_LPG,MQ4_CH4,MQ5_LPG,MQ5_CH4,LIG");

}

void loop(){

  unsigned long timestamp;
  float MQ2_alcohol; float MQ2_H2; float MQ2_Propane;
  float MQ4_LPG;
  float MQ4_CH4;
  float MQ5_LPG;
  float MQ5_CH4;

  float LIG_value;


  for (int i = 0; i < NUM_SAMPLES; i++) {

    timestamp = millis();

    // temp&humi - DHT
    int chk = DHT.read11(DHT11_PIN);

    // MQ2 - alcohol
    MQ2.update();
    MQ2.setA(3616.1); MQ2.setB(-2.675); 
    MQ2_alcohol = MQ2.readSensor();
    
    MQ2.update();
    MQ2.setA(987.99); MQ2.setB(-2.162); 
    MQ2_H2 = MQ2.readSensor();    
    
    MQ2.update();
    MQ2.setA(658.71); MQ2.setB(-2.168); 
    MQ2_Propane = MQ2.readSensor();

    // MQ4 - LPG
    MQ4.update();
    MQ4.setA(3811.9); MQ4.setB(-3.113); 
    MQ4_LPG = MQ4.readSensor(); 
    
    // MQ4 - CH4
    MQ4.setA(1012.7); MQ4.setB(-2.786); 
    MQ4_CH4 = MQ4.readSensor(); 

    // MQ5 - LPG
    MQ5.update();
    MQ5.setA(80.897); MQ5.setB(-2.431); 
    MQ5_LPG = MQ5.readSensor(); 
    
    // MQ5 - CH4
    MQ5.setA(177.65); MQ5.setB(-2.56); 
    MQ5_CH4 = MQ5.readSensor(); 

    
    // LIG
    LIG_value = analogRead(A1);



    // Print CSV data with timestamp
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(DHT.temperature);
    Serial.print(",");
    Serial.print(DHT.humidity);
    Serial.print(",");
    Serial.print(MQ2_alcohol);
    Serial.print(",");
    Serial.print(MQ2_H2);
    Serial.print(",");
    Serial.print(MQ2_Propane);
    Serial.print(",");
    Serial.print(MQ4_LPG);
    Serial.print(",");    
    Serial.print(MQ4_CH4);
    Serial.print(",");    
    Serial.print(MQ5_LPG);
    Serial.print(",");    
    Serial.print(MQ5_CH4);
    Serial.print(",");  
    Serial.print(LIG_value);
    Serial.println();

    // Wait for the next sample
    while (millis() < timestamp + SAMPLING_PERIOD_MS);

  }

  Serial.println(); //empty line between samples


}

float MQ_setup_calib(MQUnifiedsensor MQ, String name, int regression_method, float RsR0CleanAir){
  float calcR0_MQ = 0;
  Serial.print("Calibrating please wait");
  MQ.setRegressionMethod(regression_method);
  MQ.init();

  for(int i = 1; i<=10; i ++)
  {
    MQ.update(); // Update data, the arduino will read the voltage from the analog pin
    calcR0_MQ += MQ.calibrate(RsR0CleanAir);
    Serial.print(".");
  }

  if(isinf(calcR0_MQ)) {Serial.println("Warning: Conection issue, "); Serial.print(name);Serial.println(" R0 is infinite (Open circuit detected)"); while(1);}
  if(calcR0_MQ == 0){Serial.println("Warning: Conection issue, "); Serial.print(name);Serial.println(" R0 is zero (Analog pin shorts to ground)"); while(1);}
  
  Serial.print(name);Serial.println(" calibration is done.");
  return calcR0_MQ;

}