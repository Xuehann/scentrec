import re
import os
import csv
import requests
import time
import random
from bs4 import BeautifulSoup

# ========== SETUP ==========
LOG_FILE = 'logs/scrape_log.txt'
os.makedirs('data/raw_html', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def log(msg):
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

# ========== GET HTML ==========
def get_html(url, retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://www.nosetime.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for attempt in range(retries):
        try:
            response = requests.get('https://www.nosetime.com' + url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                log(f"[403 Forbidden] Blocked on: {url}")
            else:
                log(f"[{response.status_code}] Error: {url}")
        except Exception as e:
            log(f"[!] Request failed ({attempt+1}/{retries}) on {url}: {e}")
        time.sleep(random.uniform(4, 6))
    return None

# ========== SCRAPE PERFUME HTML ==========
def scrape_perfume_page(perfume_urls):
    saved_ids = set(f[:-5] for f in os.listdir('data/raw_html'))
    success, skipped, failed = 0, 0, 0

    for count, url in enumerate(perfume_urls):
        match = re.search(r'/(\d+)-', url)
        if not match:
            log(f"[!] Invalid URL: {url}")
            continue
        file_id = match.group(1)
        file_path = f'data/raw_html/{file_id}.html'
        if file_id in saved_ids:
            skipped += 1
            continue

        response = get_html(url)
        if response and response.status_code == 200:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                success += 1
                log(f"[✓] Saved: {file_id}.html")
            except Exception as e:
                log(f"[!] Error saving {file_id}.html: {e}")
                failed += 1
        else:
            failed += 1
            log(f"[!] Failed to retrieve: {url}")

        if count % 100 == 0:
            log(f"[{count}] Saved: {success}, Skipped: {skipped}, Failed: {failed}")

    log(f"✅ DONE: Total Saved: {success}, Skipped: {skipped}, Failed: {failed}")

# ========== GET URL LIST ==========
def get_url_list(filename):
    with open(filename, newline='') as f:
        return [line.strip().split('\r\n')[0] for line in f.readlines() if line.strip()]

# ========== OPTIONAL FULL SCRAPER BELOW (not needed if retrying) ==========
def get_brand_urls():
    lst = ['/pinpai/2-a.html','/pinpai/3-b.html','/pinpai/4-c.html',
           '/pinpai/5-d.html','/pinpai/6-e.html','/pinpai/7-f.html',
           '/pinpai/8-g.html','/pinpai/9-h.html','/pinpai/10-i.html',
           '/pinpai/11-j.html','/pinpai/12-k.html','/pinpai/13-i.html',
           '/pinpai/14-m.html','/pinpai/15-n.html','/pinpai/16-o.html',
           '/pinpai/17-p.html','/pinpai/18-q.html','/pinpai/19-r.html',
           '/pinpai/20-s.html','/pinpai/21-t.html','/pinpai/22-u.html',
           '/pinpai/23-v.html','/pinpai/24-w.html','/pinpai/25-x.html',
           '/pinpai/26-y.html','/pinpai/27-z.html']
    brand_urls = []
    brand_names = {}
    for count, url in enumerate(lst):
        response = get_html(url)
        if not response:
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        result = soup.find_all('a', {'class': 'imgborder'})
        for r in result:
            brand_urls.append(r.attrs['href'])
            name = r.next_sibling.text
            split = re.split(r'([a-zA-Z]+)', name)
            brand_names[split[0]] = ''.join(split[1:])
        time.sleep(5)
        log(f"Scraped {count+1} brand pages...")
    return brand_urls, brand_names

def scrape_first_page(brand_urls, range_start, range_end):
    for count, url in enumerate(brand_urls[range_start:range_end]):
        response = get_html(url)
        if not response:
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        perfume = soup.find_all('a', {'class': 'imgborder'})
        with open('data/perfumes_2.csv','a') as resultFile:
            wr = csv.writer(resultFile)
            for p in perfume:
                wr.writerow([p.attrs['href']])
        time.sleep(10)
        if count % 10 == 0:
            log(f"Scraped {count} page urls...")
    log("Finished writing perfume urls.")

if __name__ == '__main__':
    print("Import and call functions manually.")
