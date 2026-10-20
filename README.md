# YOLO + Visual Jev 檢測初步方案

這個工具以 Qwen3-VL-4B-Instruct 與原作者發布的 Visual Jev answer-supervised LoRA adapter，對圖片及指定選項進行判斷。可選用 YOLO 偵測人物，逐人裁切後判斷；也可使用 `--base-only` 跳過 adapter。

- 未加 `--use-yolo`：直接判斷整張圖片，不載入 YOLO。
- 加上 `--use-yolo`：偵測到人物時逐人裁切並輸出結果；沒有偵測到人物時改用整張圖片。
- 啟用 YOLO 時預設在 CPU 執行，Visual Jev 在 CUDA 執行，減少顯示記憶體競爭。
- 多問題模式對每張裁切圖只編碼一次視覺內容與共同前綴，並用 KV 快取批次處理問題後綴；單題維持獨立推論。

## Visual Jev 的決策流程

**LoRA 不是在 softmax 之後才強化結果。** `guanxuyu/visual-jev-4b-answer-sft` 是原作者已完成訓練的參數增量；本專案的 `quickstart.py` 只載入它進行推論，不會重新訓練或更新權重。

```text
原作者的訓練階段：圖片＋問題＋選項＋正確答案
  → 在固定的 Answer: 位置，以正確選項字母為目標訓練 LoRA

本專案的推論階段：載入 Qwen3-VL-4B-Instruct＋已訓練的 LoRA
  → 輸入圖片、問題及其候選選項
  → LM Head 產生詞彙 logits，取出 A、B、C… 等候選字母的 logits
  → 對該題的候選 logits 做 softmax，輸出選項分布與最高分選項
```

LoRA 強化的是模型產生決策分數的能力：原作者凍結視覺塔，在語言層加入可訓練的低秩參數，以 `Answer:` 位置的**完整詞彙下一 token 交叉熵**學習正確選項字母；不是用本專案顯示的候選 softmax 機率再訓練一次。詳細訓練設定見[原作者的 adapter 模型卡](https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft#training)。本專案的對應程式分別是 [`vdm_model.py`](code/vdm/models/vdm_model.py)（載入 adapter、計算 LM Head 候選 logits）與 [`quickstart.py`](code/examples/quickstart.py)（候選 softmax 與輸出）。

多問題模式下，每題有自己的問題、候選答案和 softmax 分布；圖片與共同前綴則共用快取。這些機率只表示**所提供候選答案之間的相對分數**，不是經校準的真實正確率。PPE 警報應以實際場域資料驗證，不能直接把高機率當作安全保證。

## 已測試環境

以下是本專案實際運行過的電腦與虛擬環境規格，供重現與選擇相容套件時參考；其他硬體配置尚未在本專案驗證。

| 項目 | 規格或版本 |
| --- | --- |
| 作業系統 | Windows 11 |
| 處理器 | Intel Core i7-13800H（第 13 代） |
| 系統記憶體 | 約 16 GiB |
| 顯示卡 | NVIDIA RTX A1000 Laptop GPU，6 GiB VRAM |
| NVIDIA 驅動程式 | 538.18 |
| Python | 3.12.10（`.venv`） |
| PyTorch / torchvision | 2.7.1+cu118 / 0.22.1+cu118；CUDA 執行環境 11.8 |
| 主要套件 | Transformers 5.17.0、Ultralytics 8.4.182、bitsandbytes 0.50.2、Accelerate 1.15.0、PEFT 0.21.0 |


## 環境建置、安裝

在 Windows PowerShell 中：

```powershell
cd D:\Vision_Jev\Visual-Jev
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 以下是上述已測試環境的 PyTorch 版本；其他 CUDA 環境請依 PyTorch 官方指引選擇。
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r code\requirements.txt
```

## 使用方式 (互動式提問)

```powershell
python code\examples\quickstart.py `
  --image assets\figures\demo_bottle.jpg `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --adapter guanxuyu/visual-jev-4b-answer-sft `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --interactive
```

未提供 `--request-file` 時，程式預設進入互動模式；`--interactive` 仍可明確指定。

## 多問題回答 (n=2)

```powershell
python code\examples\quickstart.py `
  --image assets\figures\demo_bottle.jpg `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --adapter guanxuyu/visual-jev-4b-answer-sft `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --request-file code\examples\bottle_questions.json `
  --question-batch-size 2
```

### 可選功能與參數

| 功能 | 指令選項 | 預設行為與說明 |
| --- | --- | --- |
| 整張圖判斷 | 不加 `--use-yolo` | 直接將完整圖片送入 Visual Jev；不載入 YOLO。 |
| YOLO 逐人判斷 | `--use-yolo` | 偵測人物並逐人裁切；未偵測到人物時改用全圖。可另設 `--yolo-model`、`--yolo-confidence`、`--yolo-padding`、`--yolo-device`。 |
| 互動提問 | `--interactive`（無 `--request-file` 時為預設） | 同一張圖可連續輸入多題；下一題直接按 Enter 才開始推論。只有一題時走原本的單題路徑，多題時共用前綴快取。 |
| JSON 多問題 | `--request-file code\examples\bottle_questions.json` | 從檔案載入多題並共用圖片編碼與前綴 KV 快取；可加 `--interactive` 在批次完成後繼續提問。 |
| 問題批次大小 | `--question-batch-size 2` | 預設 `2`；適用於互動或 JSON 的多題請求。設為 `1` 仍共用前綴，但逐題執行後綴，可降低顯示記憶體需求。 |
| 4B 基礎模型 | `--model Qwen/Qwen3-VL-4B-Instruct` | 預設值，可省略；搭配下列 4B adapter。 |
| Visual Jev adapter | `--adapter guanxuyu/visual-jev-4b-answer-sft` | 預設值，可省略；載入原作者訓練的 4B LoRA 權重。 |
| 不載入 adapter | `--base-only` | 僅用 Qwen 基礎模型。若改用 2B，須同時指定 `--model Qwen/Qwen3-VL-2B-Instruct --base-only`；**4B adapter 不能套用在 2B 模型上。** |
| 量化與圖片大小 | `--load-in-4bit`、`--max-pixels 32768` | 4-bit 量化需 CUDA；`--max-pixels` 預設為 `32768`，數值越高通常越耗記憶體。 |
| 推論裝置 | `--device cuda` | 預設 `auto`，也可指定 `cuda` 或 `mps`。 |

互動模式中，同一張圖片可連續輸入問題與逗號分隔的選項：

```text
Image [assets\figures\demo_bottle.jpg]:
Question: Is there a bottle in the image?
Choices (comma-separated): yes, no
Next question [Enter to run]: What color is the bottle?
Choices (comma-separated): green, blue, yellow, black
Next question [Enter to run]:
```

最後一行按 Enter，兩題會一起推論並共用前綴；只有一題也要在「Next question」按 Enter 才執行。下一回合的圖片欄按 Enter 可沿用圖片，在第一題留空則結束程式。模型在互動期間只載入一次，但不同回合會重新處理圖片，不跨回合保留前綴快取。

### 多問題共用前綴的呼叫方式

用 `--request-file` 載入 Visual Jev 的具名問題 JSON 格式；[bottle_questions.json](code/examples/bottle_questions.json) 含兩題及共用的 `state`。同一張全圖（若啟用 YOLO，則是每張人物裁切圖）會共用一次圖片編碼及前綴 KV 快取。以下命令會執行兩題後直接結束：

```powershell
python code\examples\quickstart.py `
  --image assets\figures\demo_bottle.jpg `
  --request-file code\examples\bottle_questions.json `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --adapter guanxuyu/visual-jev-4b-answer-sft `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --question-batch-size 2
```

需要逐人執行這組問題時，在命令末尾加上 `--use-yolo`；需要批次完成後繼續互動提問時，加上 `--interactive`。共享前綴只發生在同一次多問題請求中，不會跨互動回合保留。批次大小與效能依硬體、題數及圖片而異，不能直接套用原論文的加速倍率。

### PPE 檢查題組

[ppe_questions.json](code/examples/ppe_questions.json) 包含安全帽、反光安全背心、護目鏡、手套與安全鞋五題。題目使用英文，以符合原作者 adapter 的訓練語言；每題的 `yes` 代表至少一人明確未穿戴，`no` 代表所有可見人物都明確穿戴，`unclear` 代表影像不足以判斷，`no person visible` 代表畫面沒有人。將下列圖片路徑換成實際現場照片：

```powershell
python code\examples\quickstart.py `
  --image .\path\to\site.jpg `
  --request-file code\examples\ppe_questions.json `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --question-batch-size 2 `
  --use-yolo
```

啟用 YOLO 時會對每個偵測到的人分別執行五題；沒偵測到人則依既有邏輯改用全圖。**目前程式只輸出各題選項，不會自動彙總 NG 或發出警報。** `unclear` 應交由人工複核；安全鞋等細節通常需要足夠清晰、完整的影像。

程式會優先讀取 `models\Qwen3-VL-4B-Instruct` 與 `models\visual-jev-4b-answer-sft` 的本機目錄；不存在時，會透過 Hugging Face 下載。`models/` 已加入 `.gitignore`，不會提交大型權重。候選答案的 softmax 數值是相對於本次提供選項的分數，**不是經校準的安全告警信心度**。

## YOLO 選項

YOLO 為選用功能，只有加上 `--use-yolo` 才會載入。預設使用 `yolo11n.pt`。權重未包含在儲存庫中，Ultralytics 會在首次執行時自動下載並快取；後續執行會沿用本機檔案。信心門檻為 `0.25`，人物框周圍保留 `5%`：

```powershell
--yolo-model yolo11n.pt
--yolo-confidence 0.25
--yolo-padding 0.05
--yolo-device cpu
```

## 專案內容

```text
assets/figures/                    測試圖片
code/examples/quickstart.py        唯一執行入口
code/examples/bottle_questions.json 多問題請求範例
code/examples/ppe_questions.json    PPE 檢查題組範例
code/vdm/inference/person_crop.py  YOLO 人物偵測與裁切
code/vdm/models/                   Visual Jev 選項評分
code/tests/                        裁切測試
code/requirements.txt              Python 相依套件
```

## 模型來源與授權

| 元件 | 本專案用途 | 官方來源 | 授權 |
| --- | --- | --- | --- |
| Qwen3-VL-4B-Instruct | 預設視覺語言基礎模型 | [Qwen 4B 模型頁](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 依模型頁授權條款 |
| Visual Jev 4B answer-supervised adapter | 預設載入的 LoRA 後訓練權重；須搭配上列 4B 基礎模型 | [原作者模型卡](https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft) | 模型卡標示 Apache-2.0 |
| Qwen3-VL-2B-Instruct | 僅供 `--base-only` 的較低記憶體替代模式 | [Qwen 2B 模型頁](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 依模型頁授權條款 |
| Ultralytics YOLO11n (`yolo11n.pt`) | 偵測 `person` 類別並提供裁切座標 | [YOLO11 官方文件](https://docs.ultralytics.com/models/yolo11/) | 官方文件列出 AGPL-3.0 與 Enterprise 授權方案 |
| Visual Jev | 候選選項的 prompt 與語言模型輸出層評分方式之程式來源 | [原始碼儲存庫](https://github.com/guanxuyu-sv/Visual-Jev) · [論文](https://arxiv.org/abs/2609.25845) | 原始碼為 Apache-2.0；本專案保留其 [LICENSE](LICENSE) |

原作者的 adapter 是以 Qwen3-VL-4B-Instruct 進行答案監督後訓練；本專案在推論時載入該權重，並未自行訓練。本專案現在支援原論文的視覺前綴 KV 共用與問題後綴批次推論，但不包含其完整訓練或評測流程。原作者的公開效能數據依其提示格式、精度及評測流程取得；本專案的 YOLO 裁切、4-bit 量化與較低的 `--max-pixels` 設定，不能直接視為重現其論文結果。下載與使用各元件時須遵守各自授權條款。
