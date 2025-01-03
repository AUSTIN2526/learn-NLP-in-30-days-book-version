﻿# Ch.02 用數學來告訴你何謂神經網路
章節難度: ★★★★☆
## 內容簡介
在本章中，我將用詞嵌入層與線性分類器來進行完整的數學證明，以解釋神經網路的運作過程。這個過程包含了線性與非線性之間的轉換，並討論該如何使用偏微分找到模型需要變更的超參數，理解這個過程，將有助於我們在後續的章節中明白不同模型中的架構。

1. #### 前向傳播與激勵函數：用數學原理深入探討神經網路的前向傳播機制，結合詞嵌入層與線性分類器的操作，並理解使用激勵函數的必要性、特性及其用途。
2. #### 損失值與反向傳播：詳細解析損失函數的計算方法，逐步證明反向傳播演算法如何找出模型超參數的梯度，並說明如何透過梯度下降法調整神經網路權重，以最小化損失值。
3. #### 手刻神經網路：在本章的最後，我們將教你如何從零開始實現一個簡單的神經網路，你將學到如何將先前證明的數學公式實現在程式碼中，建立一個可以分辨情緒的語言模型，如此你將能夠理解模型實際上在做什麼，以及如何將數學公式轉換成程式碼。

## 額外教材推薦
* 無

## 修改紀錄
* 無
