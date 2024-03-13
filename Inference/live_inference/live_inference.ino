

/* Includes ---------------------------------------------------------------- */
#include <EE4002D_deployment_inferencing.h>

#include <dht.h>
#include <MQUnifiedsensor.h>
//#include <sensor_data.h>

/* Constant defines -------------------------------------------------------- */
//#define CONVERT_G_TO_MS2    9.80665f
//#define MAX_ACCEPTED_RANGE  2.0f        // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead
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

#define NUM_SAMPLES EI_CLASSIFIER_RAW_SAMPLE_COUNT //9
#define READINGS_PER_SAMPLE EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME // 1

#define SAMPLING_FREQ_HZ 2                         // Sampling frequency (Hz)
#define SAMPLING_PERIOD_MS 1000 / SAMPLING_FREQ_HZ   // Sampling period (ms)

float MQ_setup_calib(MQUnifiedsensor MQ, String name, int regression_method, float RsR0CleanAir);

// Preprocessing constants (drop the timestamp column)
float mins[] = {
  	27.0,54.0,2.88,3.07,2.01,10.6,5.22,0.15,0.23,79.0
};

float ranges[] = {
  4.0, 39.0, 5.12, 3.9, 2.57, 26.73, 10.9, 0.7, 1.24, 35.0
};

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

/**
* @brief      Arduino setup function
*/
void setup()
{
    // put your setup code here, to run once:
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

    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");



    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 9) {
        ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
        return;
    }
}


float ei_get_sign(float number) {
    return (number >= 0.0) ? 1.0 : -1.0;
}


void loop()
{
    ei_printf("\nStarting inferencing in 2 seconds...\n");

    delay(2000);

    ei_printf("Sampling...\n");

    float MQ2_alcohol; float MQ2_H2; float MQ2_Propane;
    float MQ4_LPG;
    float MQ4_CH4;
    float MQ5_LPG;
    float MQ5_CH4;
    float LIG_value;

    unsigned long timestamp;
    static float raw_buf[NUM_SAMPLES * READINGS_PER_SAMPLE];
    static signal_t signal;
    float temp;
    int max_idx = 0;
    float max_val = 0.0;
    char str_buf[40];

    for (int i = 0; i < NUM_SAMPLES; i++) {
        // Take timestamp so we can hit our target frequency
        timestamp = millis();

        // Read sensors
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


        // Store raw data into the buffer
        raw_buf[(i * READINGS_PER_SAMPLE) + 0] = DHT.temperature;
        raw_buf[(i * READINGS_PER_SAMPLE) + 1] = DHT.humidity;
        raw_buf[(i * READINGS_PER_SAMPLE) + 2] = MQ2_alcohol;
        raw_buf[(i * READINGS_PER_SAMPLE) + 3] = MQ2_H2;
        raw_buf[(i * READINGS_PER_SAMPLE) + 4] = MQ2_Propane;
        raw_buf[(i * READINGS_PER_SAMPLE) + 5] = MQ4_LPG;
        raw_buf[(i * READINGS_PER_SAMPLE) + 6] = MQ4_CH4;
        raw_buf[(i * READINGS_PER_SAMPLE) + 7] = MQ5_LPG;
        raw_buf[(i * READINGS_PER_SAMPLE) + 8] = MQ5_CH4;


        // Perform preprocessing step (normalization) on all readings in the sample
        for (int j = 0; j < READINGS_PER_SAMPLE; j++) {
            temp = raw_buf[(i * READINGS_PER_SAMPLE) + j] - mins[j];
            raw_buf[(i * READINGS_PER_SAMPLE) + j] = temp / ranges[j];
        }

        // Wait just long enough for our sampling period
        while (millis() < timestamp + SAMPLING_PERIOD_MS);
    }

    // Print out our preprocessed, raw buffer
    #if DEBUG
    for (int i = 0; i < NUM_SAMPLES * READINGS_PER_SAMPLE; i++) {
        Serial.print(raw_buf[i]);
        if (i < (NUM_SAMPLES * READINGS_PER_SAMPLE) - 1) {
        Serial.print(", ");
        }
    }
    Serial.println();
    #endif

    // Turn the raw buffer in a signal which we can the classify
    int err = numpy::signal_from_buffer(raw_buf, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
        ei_printf("ERROR: Failed to create signal from buffer (%d)\r\n", err);
        return;
    }

    // Run inference
    ei_impulse_result_t result = {0};
    err = run_classifier(&signal, &result, DEBUG_NN);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERROR: Failed to run classifier (%d)\r\n", err);
        return;
    }

    // Print the predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)\r\n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("\t%s: %.3f\r\n", 
                result.classification[i].label, 
                result.classification[i].value);
    }

    // Print anomaly detection score
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("\tanomaly acore: %.3f\r\n", result.anomaly);
    #endif

    // Find maximum prediction
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_val) {
        max_val = result.classification[i].value;
        max_idx = i;
        }
    }

    // // Print predicted label and value to LCD if not anomalous
    // tft.fillScreen(TFT_BLACK);
    // if (result.anomaly < ANOMALY_THRESHOLD) {
    //     tft.drawString(result.classification[max_idx].label, 20, 60);
    //     sprintf(str_buf, "%.3f", max_val);
    //     tft.drawString(str_buf, 60, 120);
    // } else {
    //     tft.drawString("Unknown", 20, 60);
    //     sprintf(str_buf, "%.3f", result.anomaly);
    //     tft.drawString(str_buf, 60, 120);
    // }
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER
#error "Invalid model for current sensor"
#endif
