import os
import re
import csv
import time
import random
import requests
from bs4 import BeautifulSoup

# ========== GET HTML COMMENT PAGE ==========
def get_html_comment(perfume_id, retries=3):
    url = f'https://www.nosetime.com/itmcomment.php?id={perfume_id}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://www.nosetime.com/',
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                print(f"[403 Forbidden] Blocked on ID: {perfume_id}")
                return None
            else:
                print(f"[{response.status_code}] Error on ID: {perfume_id}")
        except Exception as e:
            print(f"[!] Exception on ID {perfume_id}: {e}")
        time.sleep(random.uniform(3, 5))
    return None

# ========== SCRAPE USER RATINGS ==========
def scrape_user_ratings(perfume_ids):
    os.makedirs('data/ratings', exist_ok=True)
    with open('data/ratings/perfume_ratings.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['perfume_id', 'user_id', 'rating', 'short_comment'])
        writer.writeheader()

        for count, pid in enumerate(perfume_ids):
            response = get_html_comment(pid)
            if not response:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            comments = soup.find_all('div', {'class': 'commentbox'})

            if not comments:
                continue  # no output if no reviews

            for c in comments:
                uid_tag = c.find('div', {'class': 'userid'})
                score_tag = c.find('div', {'class': 'score'})
                text_tag = c.find('div', {'class': 'shorttext'})

                if not uid_tag or not score_tag or not text_tag:
                    continue

                user_id = uid_tag.text.strip()
                rating_match = re.search(r'(\d+)', score_tag.text)
                rating = int(rating_match.group(1)) if rating_match else None
                comment = text_tag.text.strip()

                writer.writerow({
                    'perfume_id': pid,
                    'user_id': user_id,
                    'rating': rating,
                    'short_comment': comment
                })

            if count % 50 == 0:
                print(f"[{count}] Scraped reviews for perfume ID: {pid}")

# ========== GET LIST OF PERFUME IDs ==========
def get_id_list(filename):
    with open(filename) as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ========== ENTRY POINT ==========
if __name__ == '__main__':
    perfume_ids = get_id_list('data/rated_perfume_id.csv')
    scrape_user_ratings(perfume_ids)
