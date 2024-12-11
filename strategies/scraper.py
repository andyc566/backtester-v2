from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# Set up Selenium options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Define the URL to scrape
url = "https://my.tccustomerexpress.com/"

# Initialize WebDriver
service = Service("NULL")  # Update with your ChromeDriver path
browser = webdriver.Chrome(service=service, options=chrome_options)

# Open the website
browser.get(url)

time.sleep(5)  # Allow time for the page to load

# Extract planned maintenance data
maintenance_list = []

# Example extraction - Adjust selectors based on actual site structure
try:
    maintenance_items = browser.find_elements(By.CLASS_NAME, "maintenance-item")
    for item in maintenance_items:
        receipt_point = item.find_element(By.TAG_NAME, "h3").text
        date = item.find_element(By.CLASS_NAME, "maintenance-date").text
        details = item.find_element(By.CLASS_NAME, "maintenance-details").text

        maintenance_list.append({
            "Receipt Point": receipt_point,
            "Date": date,
            "Details": details
        })

    # Convert to DataFrame
    df = pd.DataFrame(maintenance_list)

    # Export to Excel
    output_file = "planned_maintenance.xlsx"
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Data exported to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    browser.quit()
