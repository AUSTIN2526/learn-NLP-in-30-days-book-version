from flask import Flask, request, abort
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from env import get_env, load_csv_data
from gpt_response import get_response
from few_shot_selector import select_few_shot

# 主程式位址
app = Flask(__name__)
configuration, handler, client = get_env()

user = {}
prompt = '接下來的問題都要使用zh-tw回答。\n你是一個疾病管制署的客服人員，你需要根據以下的一些提示來回答用戶的問題:'
system_prompt = {"role": "system", "content": prompt}
# 載入Few-shot樣本
few_shot_examples = load_csv_data()

@app.route("/callback", methods=['POST'])
def callback():
    """
    Line機器人的Webhook入口。
    """
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    處理接收到的訊息事件。
    """
    user_id = event.source.user_id
    user[user_id] = user.get(user_id, [system_prompt])
    user[user_id].append({"role": "user", "content": event.message.text})

    # 選擇最相關的5個Few-shot樣本
    selected_few_shot = select_few_shot(few_shot_examples, event.message.text)
    # 將選擇的Few-shot樣本加入系統提示
    few_shot_prompt = "\n".join(selected_few_shot)
    user[user_id][0]["content"] = f"{prompt}\n" + few_shot_prompt
    print(few_shot_prompt)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        reply_text = get_response(client, user[user_id])
        user[user_id].append({"role": "system", "content": reply_text})

        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

if __name__ == "__main__":
    app.run(port=80, debug=True)
