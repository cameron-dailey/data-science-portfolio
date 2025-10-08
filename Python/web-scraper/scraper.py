# scraper.py
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

HTML_PATH = Path("site_mock.html")
OUT = Path("data/products.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

html = HTML_PATH.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

rows = []
for p in soup.select(".product"):
    title = p.select_one(".title").get_text(strip=True)
    price_text = p.select_one(".price").get_text(strip=True).replace("$","")
    rating_text = p.select_one(".rating").get_text(strip=True)
    rows.append({
        "title": title,
        "price": float(price_text),
        "rating": float(rating_text)
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} products to {OUT}")