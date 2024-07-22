from linebot.v3.messaging import Configuration
from linebot.v3 import WebhookHandler
from openai import AzureOpenAI
import pandas as pd

def get_env():
    # Linebot 設定
    configuration = Configuration(access_token='LINE_ACCESS_TOKEN')
    handler = WebhookHandler('LINE_CHANNEL_SECRET')
    
    # 初始化 Azure OpenAI 客戶端
    client = AzureOpenAI(
        azure_endpoint="AZURE_ENDPOINT",
        api_key=" AZURE_API_KEY",
        api_version="2023-03-15-preview"
    )
    
    return configuration, handler, client

def load_csv_data(path='CDC_chatbox.csv'):
    df = pd.read_csv(path)
    questions = '### Q:\n' + df['Question'] + '\n'
    answers = '### A:\n' + df['Answer1']
    few_shot = (questions + answers).values
    return few_shot
