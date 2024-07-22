def get_response(client, messages, model='gpt-4o'):
    """
    呼叫GPT模型來取得回應。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
