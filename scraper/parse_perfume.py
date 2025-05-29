import os
import re
import csv
from collections import defaultdict
from bs4 import BeautifulSoup

def get_attributes_from_html(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    attributes = defaultdict(list)
    for link in soup.find('ul', {'class': 'item_info'}):
        for sublink in link.find_all('a', href=True):
            href = sublink.attrs['href']
            if re.match('(/pinpai/)', href): attributes['brand'] = sublink.text
            if re.match('(/xiangdiao/)', href): attributes['theme'] = sublink.text
            if re.match('(/qiwei/)', href): attributes['note'].append(sublink.text)
            if re.match('(/tiaoxiangshi/)', href): attributes['perfumer'] = sublink.text
            if re.search('(field=attrib)', href): attributes['gender'].append(sublink.text)
            if re.search('(field=tag)', href): attributes['tags'].append(sublink.text)
    attributes['perfume_id'] = re.findall(r'(/[0-9]*-)', url)[0][1:-1]
    attributes['item_name'] = soup.find('h1').text
    attributes['url'] = url
    return attributes

def get_comments_from_html(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    attributes = defaultdict(list)
    comments = soup.find_all('div', {'class': 'hfshow'})
    for discuss in comments:
        attributes['comments'].append(discuss.text)
    if comments:
        attributes['perfume_id'] = re.findall(r'(/[0-9]*-)', url)[0][1:-1]
        attributes['url'] = url
    return attributes if attributes['comments'] else None

if __name__ == '__main__':
    os.makedirs('data/parsed', exist_ok=True)
    with open('data/parsed/attributes.csv', 'w', newline='') as f1, open('data/parsed/comments.csv', 'w', newline='') as f2:
        attr_writer = csv.DictWriter(f1, fieldnames=['perfume_id', 'item_name', 'brand', 'theme', 'note', 'perfumer', 'gender', 'tags', 'url'])
        attr_writer.writeheader()
        comment_writer = csv.DictWriter(f2, fieldnames=['perfume_id', 'url', 'comment'])
        comment_writer.writeheader()

        for filename in os.listdir('data/raw_html'):
            with open(f'data/raw_html/{filename}', 'r', encoding='utf-8') as f:
                html = f.read()
            url = f"https://www.nosetime.com/{filename[:-5]}-"  # fake full url
            attr = get_attributes_from_html(html, url)
            attr_writer.writerow({
                'perfume_id': attr['perfume_id'],
                'item_name': attr['item_name'],
                'brand': attr.get('brand', ''),
                'theme': attr.get('theme', ''),
                'note': ','.join(attr.get('note', [])),
                'perfumer': attr.get('perfumer', ''),
                'gender': ','.join(attr.get('gender', [])),
                'tags': ','.join(attr.get('tags', [])),
                'url': attr['url']
            })

            comment_data = get_comments_from_html(html, url)
            if comment_data:
                for c in comment_data['comments']:
                    comment_writer.writerow({
                        'perfume_id': comment_data['perfume_id'],
                        'url': comment_data['url'],
                        'comment': c
                    })
