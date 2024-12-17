from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os 
import time
import logging
import schedule
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Set up logging
logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Selenium options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Configure download settings
options = webdriver.ChromeOptions()
download_dir = os.path.join(os.getcwd(), "downloads")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
options.add_experimental_option("prefs", prefs)

# Launch browser
service = Service("/path/to/chromedriver")
browser = webdriver.Chrome(service=service, options=options)

def download_csv():
    service = Service("/path/to/chromedriver")
    browser = webdriver.Chrome(service=service, options=options)
    try:
        # Open target URL
        browser.get("https://my.tccustomerexpress.com/#OutagesList/")
        
        # Wait until the CSV button is clickable
        wait = WebDriverWait(browser, 20)
        csv_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'CSV') or @title='CSV']"))
        )

        # Click the CSV button
        csv_button.click()
        time.sleep(5)  # Allow time for the download
        print(f"CSV downloaded to {download_dir}")
        process_csv()

    finally:
        # Close the browser
        browser.quit()

def process_csv():
    files = sorted(os.listdir(download_dir), key=lambda x: os.path.getmtime(os.path.join(download_dir, x)))
    if len(files) < 2:
        print("Not enough files to compare.")
        return

    latest_file = os.path.join(download_dir, files[-1])
    previous_file = os.path.join(download_dir, files[-2])

    df_latest = pd.read_csv(latest_file)
    df_previous = pd.read_csv(previous_file)

    # Identify new rows
    df_new_entries = pd.concat([df_latest, df_previous]).drop_duplicates(keep=False)

    if not df_new_entries.empty:
        new_file_path = os.path.join(download_dir, "new_entries.csv")
        df_new_entries.to_csv(new_file_path, index=False)
        send_email(new_file_path)
    else:
        print("No new entries found.")

def send_email(file_path):
    sender_email = "your_email@example.com"
    recipient_email = "recipient_email@example.com"
    subject = "New Outage Entries"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    with open(file_path, "r") as file:
        attachment = MIMEText(file.read(), 'csv')
        attachment.add_header(
            "Content-Disposition", f"attachment; filename={os.path.basename(file_path)}"
        )
        msg.attach(attachment)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, "your_password")
        server.send_message(msg)
    print("Email sent successfully.")

def job():
    download_csv()

# Schedule the job daily at 6AM Mountain Time
schedule.every().day.at("06:00").do(job)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()