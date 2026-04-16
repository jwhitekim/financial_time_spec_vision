import yfinance as yf
import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}

def fetch_sp500_tickers():
    # User-Agent 헤더 추가해서 가져오기
    response = requests.get(URL, headers=headers)
    table = pd.read_html(response.text)
    sp500_tickers = table[0]["Symbol"].tolist()

    print(f"종목 수: {len(sp500_tickers)}")
    print(f"처음 10개: {sp500_tickers[:10]}")

    # 2. 전체 다운로드 (시간 소요 큼)
    data = yf.download(sp500_tickers, start="2000-01-01")
    adj_close = data["Close"]

    print(f"Shape: {adj_close.shape}")
    print(f"종목 수: {adj_close.shape[1]}")
    print(f"날짜 범위: {adj_close.index[0]} ~ {adj_close.index[-1]}")

    # 3. 저장 (매번 다운로드 안 하려면)
    adj_close.to_csv("sp500_close.csv")
    print("저장 완료: sp500_close.csv") 


def load_sp500_data(file_path="sp500_close.csv"):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"데이터 로드 완료: {file_path}")
    print(f"Shape: {df.shape}")
    return df


if __name__ == "__main__":  
    # 1. S&P 500 종목 가져오기
    fetch_sp500_tickers()
    
    # 2. 저장된 데이터 로드하기
    sp500_data = load_sp500_data()