from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os
import csv
import time
import random
import re
from bs4 import BeautifulSoup

# Define the range of years to scrape
years = sorted(list(range(1980, 2026)) + [1979, 1973, 202], reverse=True)

# Base URL for open access publications filtered by year
BASE_URL = "https://research.cbs.dk/en/publications/?publicationYear={}&nofollow=true&openAccess=%2Fdk%2Fatira%2Fpure%2Fresearchoutput%2Fopenaccesspermission%2Fopen"

# Function to sanitize file names for saving
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

# Function to initialize Selenium WebDriver with Chrome
def init_driver():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode to avoid opening browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Function to get publication links for a specific year using Selenium
def get_publication_links_for_year(year):
    driver = init_driver()
    url = BASE_URL.format(year)

    print(f"Accessing {url}")
    driver.get(url)

    # Let Cloudflare verification process complete
    time.sleep(15)

    # Check page content
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    # Extract publication links
    publication_links = [a['href'] for a in soup.select("h3.title a.link")]

    driver.quit()

    if publication_links:
        return publication_links
    else:
        print(f"No publications found for year {year}. Taking screenshot...")
        os.makedirs("screenshots", exist_ok=True)
        driver.save_screenshot(f"screenshots/no_publications_{year}.png")
        return []

# Function to download PDFs from publication links and store them by year
def download_full_text_pdfs(publication_links, year):
    year_folder = f"cbs_publications_pdfs/{year}"
    os.makedirs(year_folder, exist_ok=True)

    driver = init_driver()

    for pub_url in publication_links:
        # Ensure the URL is correctly formed
        if not pub_url.startswith("http"):
            full_url = f"https://research.cbs.dk{pub_url}"
        else:
            full_url = pub_url

        print(f"Visiting: {full_url}")
        
        try:
            driver.get(full_url)
            time.sleep(10)  # Allow the page to load

            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Try to find the link with "Full Text" first
            full_text_link = soup.select_one("a.link.document-link span")

            if full_text_link and full_text_link.text.strip().lower() == "full text":
                pdf_url = full_text_link.find_parent("a")["href"]
            else:
                # If "Full Text" is not found, look for any available document link
                document_link = soup.select_one("a.link.document-link")
                if document_link:
                    pdf_url = document_link["href"]
                    print(f"Found alternative PDF link: {pdf_url}")
                else:
                    print(f"No Full Text or alternative link found for {pub_url}")
                    continue

            # Handle relative URLs
            if not pdf_url.startswith("http"):
                pdf_url = f"https://research.cbs.dk{pdf_url}"

            print(f"Downloading PDF: {pdf_url}")
            save_pdf(pdf_url, pub_url, year, driver)

        except Exception as e:
            print(f"Error accessing {full_url}: {e}")
            continue  # Skip to the next publication link

        # Introduce a random delay to avoid being blocked
        time.sleep(random.uniform(2, 5))

    driver.quit()

# Function to save PDF files in year-based subfolders with improved handling
def save_pdf(pdf_url, publication_url, year, driver):
    sanitized_name = sanitize_filename(publication_url.split("/")[-1]) + ".pdf"
    pdf_path = os.path.join(f"cbs_publications_pdfs/{year}", sanitized_name)

    try:
        time.sleep(random.uniform(3, 6))  # Add delay to avoid rate limiting

        driver.get(pdf_url)
        time.sleep(5)

        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(driver.page_source.encode('utf-8'))

        print(f"Saved: {pdf_path}")

    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")

# Function to save publication links as a text file
def save_publication_links_to_txt(publication_links, year):
    with open(f"publication_links_{year}.txt", "w", encoding="utf-8") as file:
        for link in publication_links:
            file.write(link + "\n")
    print(f"Publication links for {year} saved to publication_links_{year}.txt")

# Function to save publication links and titles to a CSV file
def save_to_csv(publication_links, year):
    with open(f"publications_{year}.csv", "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Title", "URL"])

        for link in publication_links:
            title = link.split("/")[-1].replace("-", " ").title()
            csvwriter.writerow([title, link])

    print(f"Publication links for {year} saved to publications_{year}.csv")

# Main function to iterate through years and download publications
def main():
    for year in years:
        print(f"\nScraping publications for year {year}...\n")
        publication_links = get_publication_links_for_year(year)

        if publication_links:
            print(f"Found {len(publication_links)} publications for {year}")
            save_publication_links_to_txt(publication_links, year)
            save_to_csv(publication_links, year)
            download_full_text_pdfs(publication_links, year)
        else:
            print(f"No publications found for {year}")

        # Random delay to avoid triggering protection
        time.sleep(random.uniform(10, 20))

    print("\nScraping completed.")

if __name__ == "__main__":
    main()
