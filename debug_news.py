import requests

URL = "http://localhost:8000/news"

if __name__ == "__main__":
    try:
        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for item in data[:3]:
            print(f"{item['published_at']} - {item['title']}")
    except Exception as e:
        print("Error:", e)
