# Subtitle Band Cleaner

批量处理**烧进画面、位于底部**的字幕。工具会在每个输入视频中抽帧，用 OCR 自动找出字幕的最高位置，再对整段视频裁掉或强模糊相应的底部区域。字幕长短、字号、单行或多行不同，无须为每个视频手填高度。保留原有音轨，输出 H.264 MP4。

## 安装

需要 Python 3.10+。系统没有 `ffmpeg` 时，`imageio-ffmpeg` 会提供可执行文件。OCR 使用 [RapidOCR](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/) 和 ONNX Runtime，模型随 Python 包安装。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 使用

自动检测并处理一个视频：

```bash
.venv/bin/python clean_subtitles.py input.mp4 outputs --mode crop
```

自动处理整个文件夹；`blur` 保持原始画面尺寸：

```bash
.venv/bin/python clean_subtitles.py videos outputs --mode blur
```

如果 OCR 无法检出字体很小或特殊样式的字幕，可手动指定底部高度：

```bash
.venv/bin/python clean_subtitles.py input.mp4 outputs --mode crop --bottom-percent 15
```

每个输出旁会生成同名 JSON，记录 OCR 抽样数、检出字幕的帧数和最终遮挡高度。默认每秒抽一帧，最多 600 帧；可通过 `--sample-interval` 和 `--max-samples` 调整。`--margin-pixels` 增大字幕上方的安全边距。默认不覆盖已有结果；再次运行时加 `--overwrite`。

## 原理与边界

OCR 只把画面下方的文本视为字幕候选，并要求文字行靠近底边。多行字幕会一起纳入。最终使用低分位位置确定一条整段视频固定的裁剪线，因此字幕出现或消失时，处理区域仍然存在。`crop` 会改变画面高宽比并丢掉底部画面；`blur` 保留尺寸，但被模糊的区域不能复原。画面其他位置的字幕、滚动文字或不易被 OCR 识别的样式，可能需要手动设置或另用逐帧文字遮罩方案。独立字幕轨会从输出中移除。

原始视频和处理结果不会提交到 Git 仓库。
