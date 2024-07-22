from collections import Counter

class BPE:
    def __init__(self, vocab, pad_token='<PAD>', unk_token='<UNK>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = {tuple(word): freq for word, freq in vocab.items()}
        self.tokens = set([pad_token, unk_token])
        self.token_to_id = {pad_token: 0, unk_token: 1}  # 文字轉數字
        self.id_to_token = {0: pad_token, 1: unk_token}  # 數字轉文字

    def get_stats(self):
        pairs = Counter()
        for word, freq in self.vocab.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair):
        new_token = ''.join(pair) # 將頻率最高的字元對轉成字串

        # 更新相關資料
        if new_token not in self.tokens:
            self.tokens.add(new_token)
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token

        # 合併並更新詞彙表
        query = ' '.join(pair)  # 替換的目標字元對 (有空白)
        new_vocab = {}
        for word, freq in self.vocab.items():
            word_str = ' '.join(word) # 將原組資料轉換成字串 (有空白)
            new_word_str = word_str.replace(query, new_token) # 用repalce移除目標字元對的空白
            new_word = tuple(new_word_str.split()) # 通過空白切割字元並轉換成元組(以作為字典的鍵)
            new_vocab[new_word] = freq
        self.vocab = new_vocab

    def bpe_iterate(self, num_merges):
        for _ in range(num_merges):
            pairs = self.get_stats()
            if pairs:
                best = max(pairs, key=pairs.get)
                self.merge_vocab(best)
        return self.tokens

    def __call__(self, text):
        words = text.split()
        tokenized = [] 
        for word in words:
            word = tuple(word)

            subwords = []
            while word:  # 當word還有剩餘的字元時繼續迭代
                for i in range(len(word), 0, -1):  # 從後面開始迭代，逐漸減少子詞的長度
                    subword = ''.join(word[:i])

                    if subword in self.tokens or i == 1:
                        subwords.append(subword)  # 將子詞加入子詞列表中
                        word = word[i:]  # 將已處理過的子詞從原單詞中移除
                        break

            tokenized.extend(subwords)  # 將處理完的子詞加入最終的tokenized列表中

        # 將子詞轉換成對應的ID，如果子詞不在token_to_id中，則使用unk_token的ID
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokenized]
    
    def pad_sequence(self, sequences, max_len=None, padding_value=0):
        if max_len is None:  # 設定最大長度
            max_len = max(len(seq) for seq in sequences)  # 若沒設定自動判斷
        
        padded_sequences = []
        for seq in sequences:
            # [原始文字] + [<PAD>] * 缺少的長度
            padded_seq = seq + [padding_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences