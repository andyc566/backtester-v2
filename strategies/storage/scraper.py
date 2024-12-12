from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import logging
import schedule
import threading
import xlwings as xw

# Set up logging
logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Selenium options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

def scrape_website():
    # Define the URL to scrape
    url = "https://my.tccustomerexpress.com/"

    # Initialize WebDriver
    service = Service("/path/to/chromedriver")  # Update with your ChromeDriver path
    browser = webdriver.Chrome(service=service, options=chrome_options)

    try:
        logging.info("Opening the website")
        browser.get(url)

        time.sleep(5)  # Allow time for the page to load

        logging.info("Extracting planned maintenance data")
        maintenance_list = []

        # Example extraction - Adjust selectors based on actual site structure
        maintenance_items = browser.find_elements(By.CLASS_NAME, "maintenance-item")
        logging.debug(f"Found {len(maintenance_items)} maintenance items")

        for item in maintenance_items:
            receipt_point = item.find_element(By.TAG_NAME, "h3").text
            date = item.find_element(By.CLASS_NAME, "maintenance-date").text
            details = item.find_element(By.CLASS_NAME, "maintenance-details").text

            logging.debug(f"Extracted item: {receipt_point}, {date}, {details}")
            maintenance_list.append({
                "Receipt Point": receipt_point,
                "Date": date,
                "Details": details
            })

        # Write to Excel using xlwings
        output_file = "planned_maintenance.xlsx"
        app = xw.App()
        wb = app.books.add()
        sheet = wb.sheets[0]

        # Write headers
        headers = ["Receipt Point", "Date", "Details"]
        sheet.range("A1").value = headers

        # Write data
        for i, entry in enumerate(maintenance_list, start=2):
            sheet.range(f"A{i}").value = [entry["Receipt Point"], entry["Date"], entry["Details"]]

        wb.save(output_file)
        wb.close()
        app.quit()

        logging.info(f"Data exported to {output_file}")
        print(f"Data exported to {output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

    finally:
        browser.quit()
        logging.info("Browser closed")

# Schedule the job to run every 1 minute
schedule.every(1).minute.do(scrape_website)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Run the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()
