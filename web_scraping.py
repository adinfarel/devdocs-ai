from bs4 import BeautifulSoup
import requests

with open("index.html", "r") as html:
    doc = BeautifulSoup(html, "html.parser")

# Select
sel = doc.select("h2", class_=)
print(sel)

# -------------------
#       TITLE
# -------------------
tag = doc.title
tag.string = "Toko Mainan"
print(doc.prettify())
print(tag.string)


# -------------------
#       FIND
# -------------------
find = doc.find('h2')
print(find)


# -------------------
#       FIND_ALL
# -------------------
find_all = doc.find_all('a', href=True)
print(find_all[0].get('href'))
print([r for r in find_all])

# -----------------------------------------------------------------------------
# Scraping FastAPI

# url = "https://fastapi.tiangolo.com/reference/"

# result = requests.get(url=url)
# print(result.text)

# # Find HREF
# soup    = BeautifulSoup(result.text, "html.parser")
# print(soup.prettify())
# links = []
# a_href  = soup.find_all("a", href=True)
# for a in a_href:
#     href = a.get("href")
#     if href:
#         links.append(href)
#     else:
#         print(f"Not Found")
# print(soup.title.string)
# print(links)