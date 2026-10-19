# YOLO + Visual Jev 人物判斷工具

這個精簡工具先以 YOLO 找出圖片中的人物，再將每個人物裁切交給 Visual Jev，依照互動模式輸入的問題與候選答案進行判斷。

- 偵測到一位或多位人物：逐人裁切並分別輸出結果。
- 沒有偵測到人物：自動改用整張圖片判斷。
- YOLO 預設在 CPU 執行，Visual Jev 在 CUDA 執行，減少顯示記憶體競爭。

## 安裝

在 Windows PowerShell 中：

```powershell
cd D:\Vision_Jev\Visual-Jev
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 依電腦上的 CUDA 版本安裝 PyTorch，再安裝其餘套件。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r code\requirements.txt
```

## 使用方式

```powershell
python code\examples\quickstart.py `
  --image assets\figures\demo_bottle.jpg `
  --model Qwen/Qwen3-VL-2B-Instruct `
  --base-only `
  --load-in-4bit `
  --max-pixels 32768 `
  --device cuda `
  --interactive `
  --use-yolo
```

啟動後依序輸入圖片、問題與逗號分隔的選項：

```text
Image [assets\figures\demo_bottle.jpg]:
Question: what color is the bottle?
Choices (comma-separated): green, blue, yellow, black
```

圖片欄直接按 Enter 會沿用上一張；問題留空則結束程式。

程式會優先讀取 `models\Qwen3-VL-2B-Instruct` 的本機模型；本機不存在時才會使用 Hugging Face 模型名稱下載。`models/` 已加入 `.gitignore`，不會意外提交大型權重。

## YOLO 選項

預設使用 `yolo11n.pt`，信心門檻為 `0.25`，人物框周圍保留 `5%`：

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

## 來源與授權

Visual Jev 推論概念與模型程式衍生自 [Visual Jev](https://github.com/guanxuyu-sv/Visual-Jev)，本儲存庫保留原專案的 Apache-2.0 `LICENSE`。Qwen、Ultralytics YOLO 與模型權重各自適用其上游授權條款。
