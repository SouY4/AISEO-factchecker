import nltk
from typing import List, Dict

# NLTKの文分割用データをダウンロード（初回のみ必要）
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK's 'punkt_tab' model...")
    nltk.download('punkt_tab')

def chunk_text(document: Dict[str, str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, str]]:
    """
    単一のドキュメントテキストを、指定されたサイズとオーバーラップでチャンクに分割する。

    Args:
        document: 'source'と'text'のキーを持つ辞書。
        chunk_size: 各チャンクのおおよその文字サイズ。
        chunk_overlap: チャンク間の重複文字数。

    Returns:
        チャンク化されたテキストとソースURLを含む辞書のリスト。
    """
    text = document.get('text', '')
    source = document.get('source', '')
    if not text:
        return []

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk_text = text[start_index:end_index]
        chunks.append({'source': source, 'text': chunk_text})
        
        # 次の開始位置を計算（オーバーラップを考慮）
        start_index += chunk_size - chunk_overlap
        if start_index >= len(text):
            break
            
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    テキストを文のリストに分割する。

    Args:
        text: 検証対象の記事全文。

    Returns:
        文の文字列のリスト。
    """
    return nltk.sent_tokenize(text)

if __name__ == '__main__':
    # モジュールのテスト用コード
    test_doc = {
        'source': 'http://example.com',
        'text': 'これは一番目の文です。そして、これは二番目の文です。さらに、三番目の文が続きます。' * 50
    }
    
    # チャンク化のテスト
    chunks = chunk_text(test_doc, chunk_size=100, chunk_overlap=10)
    print("--- Chunking Test ---")
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("First chunk:")
        print(chunks[0])

    # 文分割のテスト
    test_article = "AIファクトチェッカーを開発します。これは最初の文です。これは二番目の文です！うまく分割できるでしょうか？"
    sentences = split_into_sentences(test_article)
    print("\n--- Sentence Splitting Test ---")
    print(sentences)
