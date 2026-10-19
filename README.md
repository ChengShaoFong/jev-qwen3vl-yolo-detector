# YOLO + Visual Jev 人物判斷工具

這個工具以 YOLO 偵測圖片中的人物，將每位人物的裁切圖交給 Qwen3-VL-4B-Instruct 與原作者發布的 Visual Jev answer-supervised LoRA adapter，對互動輸入的問題與候選答案輸出相對分數。也可使用 `--base-only` 跳過 adapter。

- 偵測到一位或多位人物：逐人裁切並分別輸出結果。
- 沒有偵測到人物：自動改用整張圖片判斷。
- YOLO 預設在 CPU 執行，Visual Jev 在 CUDA 執行，減少顯示記憶體競爭。

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

上述硬體與套件版本是先前 **2B base-only** 模式的已測試環境；4B＋adapter 尚未在這張 6 GiB 顯示卡上完成驗證，可能發生顯示記憶體不足。

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

## 使用方式

```powershell
python code\examples\quickstart.py `
  --image assets\figures\demo_bottle.jpg `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --adapter guanxuyu/visual-jev-4b-answer-sft `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --interactive `
  --use-yolo
```

`--model` 與 `--adapter` 均有上述預設值，可以省略。若要使用未經 Visual Jev 後訓練的 4B 基礎模型，加入 `--base-only`；若要沿用先前較省記憶體的 2B 模式，改成 `--model Qwen/Qwen3-VL-2B-Instruct --base-only`。**4B adapter 不能套用在 2B 模型上。**

啟動後依序輸入圖片、問題與逗號分隔的選項：

```text
Image [assets\figures\demo_bottle.jpg]:
Question: what color is the bottle?
Choices (comma-separated): green, blue, yellow, black
```

圖片欄直接按 Enter 會沿用上一張；問題留空則結束程式。

程式會優先讀取 `models\Qwen3-VL-4B-Instruct` 與 `models\visual-jev-4b-answer-sft` 的本機目錄；不存在時，會透過 Hugging Face 下載。`models/` 已加入 `.gitignore`，不會提交大型權重。候選答案的 softmax 數值是相對於本次提供選項的分數，**不是經校準的安全告警信心度**；本專案也未實作原論文的多問題共用快取加速。

## YOLO 選項

預設使用 `yolo11n.pt`。權重未包含在儲存庫中，Ultralytics 會在首次執行時自動下載並快取；後續執行會沿用本機檔案。信心門檻為 `0.25`，人物框周圍保留 `5%`：

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

原作者的 adapter 是以 Qwen3-VL-4B-Instruct 進行答案監督後訓練；本專案在推論時載入該權重，並未自行訓練。原作者的公開效能數據依其提示格式、精度及評測流程取得；本專案的 YOLO 裁切、4-bit 量化與較低的 `--max-pixels` 設定，不能直接視為重現其論文結果。下載與使用各元件時須遵守各自授權條款。
