# FusionSpeakerModel 結構與維度解析

## 1. 子模型初始化
- **self.w2v2**
    - 類型：Wav2Vec2Speaker（可自訂）
    - 輸入：waveform，shape = (B, W)
    - 輸出：w2v2_emb，shape = (B, w2v2_embed_dim)
- **self.resnet**
    - 類型：SimAM_ResNet34_ASP（可自訂）
    - 輸入：fbank，shape = (B, T, F)
    - 輸出：resnet_emb，shape = (B, resnet_embed_dim)
- **self.ln**
    - nn.LayerNorm(embed_dim)
    - 輸入：concat 融合後的 embedding，shape = (B, embed_dim)

## 2. forward 流程與維度

### (1) 輸入格式
- batch dict: { 'key': uttid, 'feat': fbank, 'wav': waveform (optional) }
- 'feat'：shape = (B, T, F)
- 'wav'：shape = (B, W) 或 (B, 1, W)

### (2) wav 處理
- 若 batch 內有 'wav'，直接用 batch['wav']
- 若無，根據 uttid 動態讀 wav 檔
- 維度：
    - wav = (B, W)

### (3) 特徵抽取
- w2v2_emb = self.w2v2(wav)
    - 輸入：wav, shape = (B, W)
    - 輸出：w2v2_emb, shape = (B, w2v2_embed_dim)
- resnet_emb = self.resnet(feat_tensor)
    - 輸入：feat_tensor, shape = (B, T, F)
    - 輸出：resnet_emb, shape = (B, resnet_embed_dim)

### (4) 特徵融合
- combined = torch.cat([w2v2_emb, resnet_emb], dim=1)
    - 輸入：w2v2_emb (B, w2v2_embed_dim), resnet_emb (B, resnet_embed_dim)
    - 輸出：combined, shape = (B, w2v2_embed_dim + resnet_embed_dim)

### (5) LayerNorm（可選）
- combined = self.ln(combined)
    - 輸入/輸出：shape = (B, w2v2_embed_dim + resnet_embed_dim)

### (6) 回傳
- 輸出：融合後 embedding，shape = (B, w2v2_embed_dim + resnet_embed_dim)

## 3. 維度範例（以 config 設定 w2v2_embed_dim=256, resnet_embed_dim=256 為例）
- waveform: (B, W)
- fbank: (B, T, F)
- w2v2_emb: (B, 256)
- resnet_emb: (B, 256)
- combined: (B, 512)
- LayerNorm: (B, 512)

## 4. 資料流圖（含維度）

```mermaid
graph TD
        A[batch dict: key, feat (B,T,F), wav (B,W)] -->|wav| B(Wav2Vec2 分支) --> D[w2v2_emb (B,256)]
        A -->|feat| C(ResNet 分支) --> E[resnet_emb (B,256)]
        D & E --> F[Concat (B,512)] --> G[LayerNorm (B,512)] --> H[輸出 (B,512)]
```

---

如需更細節的 backbone 結構解析，請提供 backbone 具體程式碼或 log！
# FusionSpeakerModel 結構說明

## 模型總覽

FusionSpeakerModel 是一個多特徵融合的語者嵌入模型，結合了兩種不同的前端：
- Wav2Vec2（w2v2）：以 waveform 為輸入，提取語者嵌入
- ResNet（resnet）：以 fbank（梅爾頻譜）為輸入，提取語者嵌入

最終將兩者嵌入向量串接（concatenate），並可選擇性地通過 LayerNorm 輸出。

---

## 模型結構圖

```mermaid
graph TD
    A[輸入: fbank (B, T, F)] -->|ResNet| B[resnet_emb (B, D1)]
    C[輸入: wav (B, W)] -->|Wav2Vec2| D[w2v2_emb (B, D2)]
    B --> E[Concat]
    D --> E
    E --> F[LayerNorm (可選)]
    F --> G[輸出: combined (B, D1+D2)]
```

---

## 每一層詳細說明

### 1. 輸入層
- **fbank**: (batch, time, freq) 的梅爾頻譜特徵，通常 shape 為 (B, T, 80)
- **wav**: (batch, waveform_length) 的原始音訊波形，通常 shape 為 (B, W)

### 2. ResNet 前端
- 由 `get_speaker_model(resnet_type)` 產生
- 預設型號如 SimAM_ResNet34_ASP
- 輸入 fbank，輸出 resnet_emb (B, D1)
- 可載入預訓練權重

### 3. Wav2Vec2 前端
- 由 `get_speaker_model(w2v2_type)` 產生
- 預設型號如 Wav2Vec2Speaker
- 輸入 wav，輸出 w2v2_emb (B, D2)
- 可載入預訓練權重

### 4. 融合層
- 將 resnet_emb 和 w2v2_emb 在最後一維做 concat
- 得到 (B, D1+D2) 的融合嵌入

### 5. LayerNorm（可選）
- 若 use_layernorm=True，則通過 nn.LayerNorm
- 輸出 shape 不變，僅做正規化

### 6. 輸出
- 輸出 shape: (B, D1+D2)
- 可直接送入下游分類頭（如 ArcMargin、Softmax 等）

---

## forward 流程
1. 檢查輸入是否為 dict，且含 key/feat
2. 若有 'wav'，直接用 batch['wav']，否則動態讀 wav.scp
3. wav 送入 w2v2，feat 送入 resnet
4. 兩者嵌入 concat
5. （可選）LayerNorm
6. 輸出融合嵌入

---

## 參數說明
- `w2v2_type`：wav2vec2 前端型號
- `resnet_type`：resnet 前端型號
- `w2v2_args`：wav2vec2 初始化參數
- `resnet_args`：resnet 初始化參數
- `embed_dim`：最終嵌入維度
- `use_layernorm`：是否啟用 LayerNorm
- `wav_scp_path`：wav.scp 路徑（動態讀檔用）

---

## 注意事項
- 輸入 batch 必須同時有 'feat'（fbank）和 'wav'（waveform）欄位
- 支援直接 tensor 傳入或動態讀檔
- 輸出可直接用於語者分類、驗證等下游任務

---

## 參考
- [Wav2Vec2](https://arxiv.org/abs/2006.11477)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [LayerNorm](https://arxiv.org/abs/1607.06450)
