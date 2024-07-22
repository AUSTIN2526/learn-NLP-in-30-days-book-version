from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, epochs, train_loader, valid_loader, model, optimizer, device = None, scheduler=None, early_stopping = 10, save_name = 'model.ckpt'):
        # 總訓練次數
        self.epochs = epochs

        # 訓練用資料
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 優化方式
        self.optimizer = optimizer # 優化器
        self.scheduler = scheduler # 排程器(用於動態調整學習率)
        self.early_stopping = early_stopping # 防止模型在驗證集上惡化
        
        # 判斷裝置環境
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 宣告訓練用模型
        self.model = model

        # 模型儲存名稱
        self.save_name = save_name

    def train_epoch(self, epoch):
        train_loss = 0
        train_pbar = tqdm(self.train_loader, position=0, leave=True)   # 進度條
        
        self.model.train() 
        for input_datas in train_pbar:
            for optimizer in self.optimizer:
                optimizer.zero_grad() 

            input_datas = {k: v.to(self.device) for k, v in input_datas.items()} # 將資料移動到GPU上
            outputs = self.model(**input_datas) # 進行前向傳播
            loss = outputs[0] # 取得損失值
            loss.backward() # 反向傳播

            # optimizer 可能有數個
            for optimizer in self.optimizer:
                optimizer.step()

            # scheduler 可能有數個
            if self.scheduler is not None:
                for scheduler in self.scheduler:
                    scheduler.step()
            

            postfix_dict = {'loss': f'{loss.item():.3f}'} # 定義進度條尾部顯示的資料
            train_pbar.set_description(f'Train Epoch {epoch}')  # 進度條開頭
            train_pbar.set_postfix(postfix_dict)                # 進度條結尾

            train_loss += loss.item()  # 加總損失值

        return train_loss / len(self.train_loader) # 計算平均損失
    
    def validate_epoch(self, epoch):
        valid_loss = 0
        valid_pbar = tqdm(self.valid_loader, position=0, leave=True)
        
        self.model.eval()     # 將模型轉換成評估模式
        with torch.no_grad(): # 防止梯度計算
            for input_datas in valid_pbar:
                input_datas = {k: v.to(self.device) for k, v in input_datas.items()}
            
                outputs = self.model(**input_datas) 
                loss = outputs[0]
                
                valid_pbar.set_description(f'Valid Epoch {epoch}')
                valid_pbar.set_postfix({'loss':f'{loss.item():.3f}'})

                valid_loss += loss.item()

        return valid_loss / len(self.valid_loader)
    
    def train(self, show_loss=True):
        best_loss = float('inf')
        loss_record = {'train': [], 'valid': []}
        stop_cnt = 0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validate_epoch(epoch)

            loss_record['train'].append(train_loss) # 加入訓練的平均損失
            loss_record['valid'].append(valid_loss) # 加入驗證的平均損失

            # 儲存最佳的模型
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_name) # 儲存模型
                print(f'Saving Model With Loss {best_loss:.5f}')
                stop_cnt = 0
            else:
                stop_cnt += 1

            # Early stopping
            if stop_cnt == self.early_stopping:
                output = "Model can't improve, stop training"
                print('-' * (len(output) + 2))
                print(f'|{output}|')
                print('-' * (len(output) + 2))
                break

            print(f'Train Loss: {train_loss:.5f}', end='| ')
            print(f'Valid Loss: {valid_loss:.5f}', end='| ')
            print(f'Best Loss: {best_loss:.5f}', end='\n\n')
        
        # 顯示訓練曲線圖
        if show_loss:
            self.show_training_loss(loss_record)
        
    def show_training_loss(self, loss_record):
        train_loss, valid_loss = [i for i in loss_record.values()]

        plt.plot(train_loss)
        plt.plot(valid_loss)
        # 標題
        plt.title('Result')
        # Y軸座標
        plt.ylabel('Loss')
        # X軸座標
        plt.xlabel('Epoch')
        # 顯示各曲線名稱
        plt.legend(['train', 'valid'], loc='upper left')
        # 顯示曲線
        plt.show()