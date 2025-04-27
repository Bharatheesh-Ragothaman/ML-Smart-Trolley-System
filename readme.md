# ML Powered Smart Trolley System with Hybrid Recommendation System and Bill Alerts

- This project aims to enhance the user experience in the Retail environments by automating product purchase methods.
- Smart trolley is the main logic of this project where the retail store trolleys are converted into smart component, 
where we can : <br> <br>
    **1. Automate the trolley movement without human intervention** <br><br>
    **2. Integrate user and product scanning within the trolley** <br><br>
    **3. Integrate personalized product recommendations with respective rack numbers** <br><br>
    **4. Generate bill remotely from the trolley in PDF format and send to customer's whatsapp after checkout** <br><br>

## Trolley Movement ✅

- BLE enabled ESP 32 is used as Bluetooth Beacon. This will be present remotely in the hands of the customer. This beacon will send BLE signals to the Raspberry to detect the proximity of the customer.

- 3 ultrasonic sensors are used for obstacle detection. The sensors are placed in the trolley in this manner: <br>
    1. Front <br>
    2. Left <br>
    3. Right <br>

-  The combination of ultrasonic obstacle detection and BLE enabled customer proximity detection helps in automated movement of the trolley.

- Trolley movements are : Forward, Backward, Left, Right.

- The customer Proximity range is set to : -75 RSSI

- The ultrasonic sensors have various combined logics for movement. But, basically they work in the concept that, if the obstacle is less than 20 cm in either of the directions, then the trolley will stop.

- This means that the trolley stops when the customer is near.

- This movement logic works well in ideal scenarios.

## RFID Scanning ✅ 

- 13MHz RFID cards are used for User and product scanning.

- The RFID cards are mapped with products and users. This is done in the datasets.

- Separate checkout card is prepared with which the product purchasing is stopped and it is proceeded to bill generation phase.

**Customer RFID** : Verifies customer & displays welcome message.

**Product RFID**  : Checks product validity, updates cart, gets recommendations.

**Checkout RFID** : Calculates bill & sends bill alerts to customer.

## Recommendation System ✅

**Hybrid algorithm - Combination of Apriori and Random Forest.**

- Apriori provides rulesets from global purchase patterns.

- Random forest makes personalized predictions based on current customer's cart purchase patterns.

- Overall, personalized product recommendations are provided to the customer from both global and user specific patterns.

- Top 3 products are recommended from these patterns, to the customer.

- Rack numbers of the suggested products are also provided. Both suggested products as well as rack numbers are displayed on the LCD.

**Displays recommendations in this format** <br><br>
    **Bread-A1** <br><br>
    **Butter-B3**

## Bill Alerts To Customer's WhatsApp ✅ 

- Once the checkout card is scanned, Purchase bill is generated in PDF format and stored locally.

- The locally stored PDF is then uploaded to Firebase cloud service.

- The cloud stored PDF's host url is taken and shared to Twilio API services.

- The Twilio service will make API call to the customer's WhatsApp number with the PDF's host URL.

- The customer will receive Bill as PDF in the WhatsApp. 

- The Bill is generated in the below format.

### Star Hypermarket <br>
**Lifestyle at one place!** <br><br> 
**Date:** Purchase date <br>
**Time:** Purchase time <br>
**Customer:** {Customer name} <br>
**Contact:** {Customer contact} <br>

**Product Quantity  Price  Amount** <br>
Bread      2        50    100    <br>
Butter    5        40    200 <br>

Total Bill - Rs. 300 <br>

Thank you <br>
Visit Again! <br>