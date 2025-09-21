import requests
from bs4 import BeautifulSoup
from typing import List, Dict

def extract_text_from_urls(urls: List[str]) -> List[Dict[str, str]]:
    """
    URLのリストを受け取り、各URLから本文テキストを抽出する。

    Args:
        urls: 参考文献のURL文字列のリスト。

    Returns:
        ソースURLと本文テキストを含む辞書のリスト。
    """
    documents = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }

    for url in urls:
        try:
            print(f"Fetching content from: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # HTTPエラーがあれば例外を発生させる

            soup = BeautifulSoup(response.content, 'html.parser')

            # 不要なタグ（スクリプト、スタイル、ヘッダー、フッターなど）を削除
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
                element.decompose()
            
            # テキストを取得し、余分な空白や改行を整理
            body_text = soup.get_text(separator='\n', strip=True)
            
            documents.append({'source': url, 'text': body_text})
            print(f"Successfully extracted text from: {url}")

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {url}: {e}")
            
    return documents

if __name__ == '__main__':
    # モジュールのテスト用コードを実行
    test_urls = [
        "https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB"
    ]
    extracted_docs = extract_text_from_urls(test_urls)
    if extracted_docs:
        print("\n--- Extracted Content (First 200 chars) ---")
        print(f"Source: {extracted_docs[0]['source']}")
        print(extracted_docs[0]['text'][:200] + "...")