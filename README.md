# YOLO + Visual Jev 人物判斷工具

這個工具以 YOLO 偵測圖片中的人物，將每位人物的裁切圖交給 Qwen3-VL，並使用改寫自 Visual Jev 的選項評分方式，對互動輸入的問題與候選答案輸出機率。

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

程式會優先讀取 `models\Qwen3-VL-2B-Instruct` 的本機模型；本機不存在時，會透過 Hugging Face 下載。`models/` 已加入 `.gitignore`，不會提交大型權重。

## YOLO 選項

預設使用儲存庫內的 `yolo11n.pt`；若該檔案不存在，Ultralytics 會在首次執行時下載。信心門檻為 `0.25`，人物框周圍保留 `5%`：

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
| Qwen3-VL-2B-Instruct | 視覺語言基礎模型，判讀圖片與候選答案 | [Qwen 模型頁](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 模型頁標示 Apache-2.0 |
| Ultralytics YOLO11n (`yolo11n.pt`) | 偵測 `person` 類別並提供裁切座標 | [YOLO11 官方文件](https://docs.ultralytics.com/models/yolo11/) | 官方文件列出 AGPL-3.0 與 Enterprise 授權方案 |
| Visual Jev | 候選選項的 prompt 與語言模型輸出層評分方式之程式來源 | [原始碼儲存庫](https://github.com/guanxuyu-sv/Visual-Jev) · [論文](https://arxiv.org/abs/2609.25845) | 原始碼為 Apache-2.0；本專案保留其 [LICENSE](LICENSE) |

目前的固定命令使用 `--base-only`，**不會載入**原作者發布的 [Visual Jev 4B LoRA 適配器](https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft)。因此，這裡的 2B 推論結果不等同於論文中 4B 微調模型的結果。本專案沒有重新訓練 Qwen 或 Visual Jev 模型；儲存庫內的 `yolo11n.pt` 為 Ultralytics 權重，使用與再散布時須遵守其授權條款。
