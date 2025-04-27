import threading
import os
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from mfrc522 import SimpleMFRC522
import csv
from datetime import datetime
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import requests
import asyncio
from bleak import BleakScanner
import firebase_admin
from firebase_admin import credentials, storage
from twilio.rest import Client
import pytz

# Initialize Firebase
cred = credentials.Certificate('Replace with your service account file')  
firebase_admin.initialize_app(cred, {
    'storageBucket': 'Your Firebase storage bucket name'  
})

# Twilio credentials
account_sid = 'Replace with your Twilio SID'  
auth_token = 'Replace with your Twilio Auth Token'    
twilio_number = f'whatsapp:{Twilio_WhatsApp_Sandbox_Number}'  
# Initialize Twilio client globally
client = Client(account_sid, auth_token)  # Initialize Twilio client

lcd_lock = threading.Lock()

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# L298N Motor Driver Pins
IN1, IN2, IN3, IN4 = 17, 18, 22, 23
ENA, ENB = 5, 6
for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)

# === BLE Detection ===
TARGET_BLE_MAC = "A8:42:E3:47:B0:2E"  # Your ESP32 MAC address
BLE_RANGE_THRESHOLD = -75

# PWM setup
pwm_A = GPIO.PWM(ENA, 1000)
pwm_B = GPIO.PWM(ENB, 1000)
pwm_A.start(0)
pwm_B.start(0)
DEFAULT_SPEED = 50

# Shared TRIG pin
TRIG_PIN = 16
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.output(TRIG_PIN, False)

# Dictionary of ECHO pins
ECHO_PINS = {
    'left': 24,
    'center': 25,
    'right': 13
}

# Setup ECHO pins
for pin in ECHO_PINS.values():
    GPIO.setup(pin, GPIO.IN)

# RFID Reader Setup
reader = SimpleMFRC522()

# Initialize 20X4 LCD
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,        
    port=1,
    cols=20,
    rows=4,
    charmap='A00',
    auto_linebreaks=False
)

# Load CSV data
def load_csv(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row['RFID']] = row
    return data

cust_details = load_csv('/ML-Smart-Trolley-System/software/datasets/cust-details.csv')
product_details = load_csv('/ML-Smart-Trolley-System/software/datasets/product-details.csv')

# File Paths
cust_purchases_file = "/ML-Smart-Trolley-System/software/datasets/cust-past-purchases.csv"
rack_mapping_file = "/ML-Smart-Trolley-System/software/datasets/product-details.csv"

# Load datasets
customer_purchases = pd.read_csv(cust_purchases_file)
rack_mapping = pd.read_csv(rack_mapping_file)

cart = {}
customer_name, customer_phone = "", ""
user_verified = False

def set_speed(left=DEFAULT_SPEED, right=DEFAULT_SPEED):
    pwm_A.ChangeDutyCycle(left)
    pwm_B.ChangeDutyCycle(right)

# Motor Control Functions
def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed()

def move_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed()

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    set_speed()

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    set_speed()

def stop_motor():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    set_speed(0,0)

#Function to measure distance from a given ECHO pin
def measure_distance(echo_pin):
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    timeout = time.time() + 0.05
    pulse_start = None
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return 999
    pulse_end = None
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return 999

    if pulse_start is None or pulse_end is None:
        return 999

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

async def trolley_movement():
    try:
        # Step 1: BLE Scan for customer proximity
        ble_devices = await BleakScanner.discover(timeout=2.0)
        customer_rssi = None

        for d in ble_devices:
            if d.address == esp32_mac_address:
                customer_rssi = d.rssi
                break

        # Step 2: Ultrasonic distances
        front_dist = measure_distance(ECHO_PINS["center"])
        left_dist = measure_distance(ECHO_PINS["left"])
        right_dist = measure_distance(ECHO_PINS["right"])

        print(f"[Trolley Movement] RSSI: {customer_rssi} dBm, Front: {front_dist} cm, Left: {left_dist} cm, Right: {right_dist} cm")

        # Step 3: Sensor Fusion Logic
        if customer_rssi is not None and customer_rssi >= -70:
            # Customer is nearby (strong BLE signal)
            if 30 < front_dist < 100:
                print("Customer ahead — moving forward")
                move_forward()
            elif front_dist <= 30:
                print("Too close to obstacle (maybe customer). Stopping.")
                stop_motor()
                await asyncio.sleep(1)
            else:
                print("Customer nearby but not aligned — scanning surroundings")
                stop_motor()
                await asyncio.sleep(0.5)
        else:
            # No BLE signal or weak signal — Obstacle avoidance
            if front_dist < 20:
                stop_motor()
                print("Obstacle ahead. Avoiding...")
                await asyncio.sleep(0.5)

                if left_dist > right_dist and left_dist > 30:
                    turn_left()
                    await asyncio.sleep(0.5)
                elif right_dist > 30:
                    turn_right()
                    await asyncio.sleep(0.5)
                else:
                    print("No space to turn. Waiting.")
                    await asyncio.sleep(1)
                stop_motor()
            else:
                print("Path clear, moving forward")
                move_forward()

    except Exception as e:
        print(f"Error in trolley_movement: {e}")
        stop_motor()

def clear_line(line_number):
    with lcd_lock:
        lcd.cursor_pos = (line_number, 0)
        lcd.write_string(" " * 20)

def display_message(message, line_number, scroll=False, delay=0.3, timeout=None):
    with lcd_lock:
        if scroll and len(message) > 20:
            scroll_length = len(message) - 19
            for i in range(scroll_length):
                lcd.cursor_pos = (line_number, 0)
                lcd.write_string(message[i:i+20])
                time.sleep(delay)
            lcd.cursor_pos = (line_number, 0)
            lcd.write_string(message[-20:].ljust(20))
        else:
            lcd.cursor_pos = (line_number, 0)
            lcd.write_string(message.ljust(20))

    if timeout:
        time.sleep(timeout)
        clear_line(line_number)

# Load valid product names from dataset (case-insensitive matching)
valid_products = {p.lower(): p for p in customer_purchases['Past_Purchases'].unique()}  

# Preprocess purchase data for Apriori
basket = customer_purchases.groupby(['Customer_ID', 'Past_Purchases'])['Past_Purchases'].count().unstack().fillna(0)

# Convert basket to boolean type (as recommended by mlxtend)
basket = basket.astype(bool)

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.002, use_colnames=True)  
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
rules = rules.nlargest(1000, 'lift')  

# Expand rules dataset
rules_expanded = rules.explode('antecedents').explode('consequents')
rules_expanded = rules_expanded[['antecedents', 'consequents', 'confidence', 'lift']]
rules_expanded.columns = ['Product_A', 'Product_B', 'Confidence', 'Lift']

# Convert categorical values to integer encoding
product_mapping = {product: idx for idx, product in enumerate(pd.concat([rules_expanded['Product_A'], rules_expanded['Product_B']]).unique())}
rules_expanded.replace(product_mapping, inplace=True)

# Explicitly set type to avoid downcasting warning
rules_expanded = rules_expanded.astype({'Product_A': 'int32', 'Product_B': 'int32'})

# Train Random Forest model
X = rules_expanded[['Product_A', 'Confidence', 'Lift']]
y = rules_expanded['Product_B']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Updated Recommendation Function
def get_recommendations(cart):
    recommendations = set()

    # Convert cart products to encoded values
    encoded_cart = [product_mapping.get(p, -1) for p in cart]

    # Apriori-based recommendations
    matching_rules = rules_expanded[rules_expanded['Product_A'].isin(encoded_cart)]

    for rec in matching_rules.sort_values(by=['Confidence', 'Lift'], ascending=False)['Product_B'].unique():
        if rec in product_mapping.values():
            product_name = [k for k, v in product_mapping.items() if v == rec][0]
            if product_name not in cart:
                recommendations.add(product_name)

    # Random Forest-based recommendations
    for product in cart:
        if product in product_mapping:
            product_encoded = product_mapping[product]
            try:
                # Convert input to a DataFrame with column names to prevent warning
                pred = rf_model.predict(pd.DataFrame([[product_encoded, 0.5, 1.0]], columns=['Product_A', 'Confidence', 'Lift']))
                for item in pred:
                    if item in product_mapping.values():
                        product_name = [k for k, v in product_mapping.items() if v == item][0]
                        if product_name not in cart:
                            recommendations.add(product_name)
            except:
                continue

    # Ensure cart items are not suggested
    recommendations = {item.strip() for rec in recommendations for item in rec.split(",") if item.strip() not in cart}

    # Ensure exactly 3 unique recommendations
    final_recommendations = list(recommendations)[:3]

    # Fill missing recommendations with top-selling products
    if len(final_recommendations) < 3:
        top_products = customer_purchases['Past_Purchases'].value_counts().index.tolist()
        for prod in top_products:
            if prod not in cart and prod not in final_recommendations and len(final_recommendations) < 3:
                final_recommendations.append(prod)

    # Fetch rack numbers from `rack_mapping`
    recommendation_dict = {}
    for product in final_recommendations:
        rack_number = rack_mapping.loc[rack_mapping['Product'] == product, 'Rack_Number']
        recommendation_dict[product] = rack_number.values[0] if not rack_number.empty else "Unknown"

    return recommendation_dict  # Return dictionary with {product: rack_number}

def generate_invoice_pdf(customer_name, customer_phone, cart):
    now = datetime.now(ist)
    filename = f"Invoice_{customer_phone}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    filepath = os.path.join(os.getcwd(), filename)
    styles = getSampleStyleSheet()
    date_str = now.strftime('%d-%m-%Y')
    time_str = now.strftime('%H:%M:%S')

    # Define the two-line, 2-column layout
    header_data = [
        [f"CUSTOMER : {customer_name}", f"DATE : {date_str}"],
        [f"CONTACT : {customer_phone}", f"TIME : {time_str}"]
    ]

    # Prepare invoice data
    data = [["PARTICULAR", "QUANTITY", "UNIT PRICE", "TOTAL"]]
    total_cost = 0

    for item, details in cart.items():
        qty = details["quantity"]
        total = details["cost"]
        unit_price = total / qty 
        total_cost += total
        data.append([item, str(qty), f"Rs. {unit_price:.2f}", f"Rs. {total:.2f}"])

    data.append(["", "", "GRAND TOTAL", f"Rs. {total_cost:.2f}"])

    # Custom styles
    heading1_center = ParagraphStyle(name="Heading1CenterAlign", parent=styles["Heading1"], alignment=TA_CENTER)
    heading2_center = ParagraphStyle(name="Heading2CenterAlign", parent=styles["Heading2"], alignment=TA_CENTER)
    title_center = ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER)

    # Create PDF
    doc = SimpleDocTemplate(filepath, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    story = []

    # Title & Header
    # Create a 2-column table
    header_table = Table(header_data, colWidths=[250, 250])  # Adjust widths as needed
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),     # Left column
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),    # Right column
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('BOX', (0, 0), (-1, -1), 0, colors.white),       # No borders
        ('INNERGRID', (0, 0), (-1, -1), 0, colors.white), # No grid
    ]))


    # Table
    num_rows = len(data)
    bill_table = Table(data, colWidths=[180, 80, 100, 100])

    bill_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEABOVE', (0, num_rows-1), (-1, num_rows-1), 1, colors.black),
        ('LINEBEFORE', (1, 0), (1, -1), 1, colors.black),
        ('LINEBEFORE', (2, 0), (2, -1), 1, colors.black),
        ('LINEBEFORE', (3, 0), (3, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -2), 'RIGHT'),
        ('ALIGN', (3, 1), (3, -2), 'RIGHT'),
        ('ALIGN', (2, -1), (3, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (2, -1), (3, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10)
    ]))

    story.append(Paragraph("S T A R Hypermarket", title_center))
    story.append(Paragraph("Lifestyle at one place!", heading1_center))
    story.append(Paragraph("PURCHASE BILL", heading2_center))
    story.append(Spacer(1,15))
    story.append(header_table)
    story.append(Spacer(1,15))
    story.append(bill_table)
    story.append(Spacer(1, 30))
    story.append(Paragraph("THANK YOU FOR SHOPPING WITH US!", heading1_center))
    story.append(Spacer(1,15))
    story.append(Paragraph("VISIT AGAIN!", heading1_center))

    # Save PDF
    doc.build(story)
    print(f"PDF generated at: {filepath}")
    return filepath

# Upload PDF to Firebase Storage
def upload_pdf_to_firebase(pdf_path):
    bucket = storage.bucket()  # Reference to the default Firebase bucket
    blob = bucket.blob(os.path.basename(pdf_path))
    blob.upload_from_filename(pdf_path)
    blob.make_public()  # Make the file public
    print("Firebase file uploaded at:", blob.public_url)
    return blob.public_url  # Return the public URL of the file

# Send PDF via WhatsApp using Twilio
def send_whatsapp_message_with_pdf(pdf_url, recipient_number):
    country_code="+91"
    to_number = country_code + recipient_number

    message = client.messages.create(
        body="Thank you for your purchase! Please find your bill attached.",
        from_=twilio_number,  # Twilio WhatsApp number
        to=f'whatsapp:{to_number}',  # Customer's phone number
        media_url=[pdf_url]  # Link to the uploaded PDF in Firebase Storage
    )
    print(f"Message SID: {message.sid}")
    print("PDF sent to Whatsapp")
    return True

def cleanup():
    stop_motor()
    pwm_A.stop()
    pwm_B.stop()
    GPIO.cleanup()

checkout = "532913062514" # RFID card ID of Checkout card
esp32_mac_address = "A8:42:E3:47:B0:2E"

async def main():
    try:
        user_verified = False
        checkout_scanned = False
        last_scanned = None
        last_scan_time = 0

        # 1. USER VERIFICATION
        while not user_verified:
            display_message("Scan User Card First", line_number=0, scroll=True)
            print("Scan User Card First")
            cust_id, _ = reader.read()
            cust_id = str(cust_id)

            if cust_id in cust_details:
                customer_name = cust_details[cust_id]['Customer-Name']
                customer_phone = cust_details[cust_id]['Phone']
                display_message(f"Welcome {customer_name}!", line_number=2, scroll=True, timeout=5)
                print(f"Welcome {customer_name}!")
                user_verified = True

        # 2. TROLLEY MOVEMENT + PRODUCT SCANNING LOOP
        while not checkout_scanned:
            # Step A: Move trolley using BLE + Ultrasonic
            await trolley_movement()

            # Step B: After movement, re-scan BLE to decide whether to allow product scan
            ble_devices = await BleakScanner.discover(timeout=2.0)
            customer_rssi = None
            for d in ble_devices:
                if d.address == esp32_mac_address:
                    customer_rssi = d.rssi
                    break

            # Step C: Product scan only if customer is close
            if customer_rssi is not None and customer_rssi >= -70:
                stop_motor()
                display_message("Scan Product Card", line_number=0, scroll=True)
                print("Scan Product Card")
                prod_id, _ = reader.read()
                prod_id = str(prod_id)

                if prod_id == checkout:
                    stop_motor()

                    invoice_path = generate_invoice_pdf(customer_name, customer_phone, cart)
                    if os.path.exists(invoice_path):
                        invoice_url = upload_pdf_to_firebase(invoice_path)
                        success = send_whatsapp_message_with_pdf(invoice_url, customer_phone)
                        if success:
                            display_message("Invoice sent", line_number=0, timeout=3)
                            display_message("   via  WhatsApp!   ", line_number=1, timeout=3)
                            print("Invoice sent via WhatsApp!")
                        else:
                            display_message("Failed to", line_number=2, timeout=3)
                            display_message("send Invoice", line_number=3, timeout=3)
                            print("Failed to send Invoice")

                    display_message("   Thank You!   ", line_number=2, timeout=3)
                    display_message("    Visit Again!    ", line_number=3, timeout=3)
                    print("Thank You! Visit Again!")
                    checkout_scanned = True
                    break

                # Avoid duplicate scans
                current_time = time.time()
                if prod_id == last_scanned and (current_time - last_scan_time) < 3:
                    await asyncio.sleep(1)
                    continue

                # Add product to cart
                elif prod_id in product_details:
                    product = product_details[prod_id]['Product']
                    price = float(product_details[prod_id]['Price'])

                    cart.setdefault(product, {'quantity': 0, 'cost': 0})
                    cart[product]['quantity'] += 1
                    cart[product]['cost'] += price

                    display_message(f"   Added to Cart:   ", line_number=1,timeout=2)
                    display_message(f"  {product}  " , line_number=2, scroll=True,timeout=3)
                    display_message(f"     {price}  Rs     ", line_number=3,timeout=3)
                    print(f"Added to Cart: {product} - {price} Rs")

                    # Show Recommendations
                    recommendations = get_recommendations(list(cart.keys()))
                    if recommendations:
                        display_message("Your Personalized", line_number=0,timeout=2.5)
                        display_message("Product Recommendations . . .", line_number=2, scroll=True, timeout=1.5)
                        display_message("  PRODUCT  |  RACK  ", line_number=0)
                        print("Your personalized recommendations:")
                        recommendation_line_number = 1
                        for product, rack in recommendations.items():
                            display_message(f"{product} | {rack}", line_number=recommendation_line_number, scroll=True,timeout=3)
                            print(f"{product} : {rack}")
                            recommendation_line_number += 1

                    last_scanned = prod_id
                    last_scan_time = current_time
                    await asyncio.sleep(1.5)

    except KeyboardInterrupt:
        cleanup()
        print("Process Terminated")

asyncio.run(main())