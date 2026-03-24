url = "https://fastapi.tiangolo.com/reference/"

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