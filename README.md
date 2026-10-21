# Subtitle Band Cleaner

先检测视频是否有**烧进画面、位于底部**的对白字幕，再批量裁剪或模糊。有字幕时，工具根据 OCR 自动估计字幕区域，保留音轨并输出 H.264 MP4；没有检出字幕时，只重命名原文件，不重新编码，也不生成一份处理后的视频。

## 检测与命名

输入 `clip.mp4` 时，命名规则如下。**原视频会在输入目录中重命名，文件内容不变**；处理结果写入指定的输出目录。

| 检测结果 / 操作 | 输入目录中的原视频 | 输出目录中的成片 |
| --- | --- | --- |
| 未检出字幕 | `clip_no_subtitle.mp4` | 不生成 |
| 有字幕，裁剪 | `clip_with_subtitle.mp4` | `clip_subtitle_cropped.mp4` |
| 有字幕，模糊 | `clip_with_subtitle.mp4` | `clip_subtitle_blurred.mp4` |

原视频保留原扩展名。已有 `_no_subtitle` 或 `_with_subtitle` 后缀不会重复添加；批处理会跳过 `_subtitle_blurred` 和 `_subtitle_cropped` 成片。若重命名将覆盖另一个原视频，工具会报错并保留两个原文件。单个视频解码或处理失败时会继续检查同批次的其他视频，并以非零状态退出。

## 单帧效果对比

以下是同一段视频在 **9.5 秒**处的原始帧与实际处理结果。工具自动检测到需处理底部 **13.3%**：裁剪模式移除该区域，模糊模式保留原始尺寸并遮挡字幕。图片按相同宽度展示，保留各自的高宽比。

| 原始帧 | 自动裁剪 | 自动模糊 |
| --- | --- | --- |
| <img src="docs/images/frame-original.png" width="300" alt="原始帧：底部有 What? 字幕"> | <img src="docs/images/frame-crop.png" width="300" alt="裁剪结果：移除底部字幕区域"> | <img src="docs/images/frame-blur.png" width="300" alt="模糊结果：保留画面尺寸，字幕不可读"> |
| 626 × 360 | 626 × 312 | 626 × 360 |

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

只检测并重命名所有原视频，不生成裁剪或模糊成片：

```bash
.venv/bin/python clean_subtitles.py videos outputs --classify-only
```

已检出字幕时，可手动指定底部高度；这个参数本身不会让无字幕视频进入处理分支：

```bash
.venv/bin/python clean_subtitles.py input.mp4 outputs --mode crop --bottom-percent 15
```

每个视频在输出目录都有一份 JSON 检测记录，包括新原文件名、是否检出字幕、是否处理、候选文字帧数和字幕区域。无字幕的视频对应 `clip_no_subtitle.json`。默认每秒抽一帧，最多 600 帧；可通过 `--sample-interval` 和 `--max-samples` 调整。`--margin-pixels` 增大字幕上方的安全边距。默认不覆盖已有成片；再次处理时加 `--overwrite`。

默认要求同一字幕位置至少出现 3 种不同文字，以减少把台标、固定人物介绍和场景标牌误判为对白字幕。仅有一两句字幕的短视频可降低阈值；非居中字幕可扩大检测范围：

```bash
.venv/bin/python clean_subtitles.py input.mp4 outputs --min-distinct-captions 1 --alignment any
```

如果已经人工确认有字幕，但 OCR 未检出，可明确强制处理并给出高度：

```bash
.venv/bin/python clean_subtitles.py input.mp4 outputs --force-process --bottom-percent 15
```

## 原理与边界

OCR 只把画面下方、默认居中的文本视为字幕候选，检查其底线和水平位置是否稳定、文字是否随时间更换；微小 OCR 拼写差异会合并。多行字幕会一起纳入。最终使用低分位位置确定一条整段视频固定的裁剪线，因此字幕出现或消失时，处理区域仍然存在。`crop` 会改变画面高宽比并丢掉底部画面；`blur` 保留尺寸，但被模糊的区域不能复原。

`no_subtitle` 表示抽样中未检出符合条件的底部对白字幕，并非逐帧证明完全没有文字。很稀疏、短暂、特殊样式或其他位置的字幕可能漏检；变化的节目横幅也可能被误认。仅有独立字幕轨的视频不在本工具的画面 OCR 检测范围内。进入处理分支时，独立字幕轨会从输出中移除；无字幕分支不会改动任何视频字节或轨道。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖无字幕不调用编码器、重命名前后内容哈希一致、两种处理后缀、重复运行、原文件名冲突、解码失败和批处理继续执行，以及固定文字与变化字幕的区分。

2026-10-21 本地样本验证使用默认参数：约每秒抽一帧、每视频最多 600 帧、同一位置至少 3 种不同字幕。新增文件夹按 6 个不同文件名取样（共 9 个文件条目），并与此前 5 段有字幕视频一起测试。

| 样本组 | 数量 | 检测结果 | 原视频改名后 SHA-256 | 生成成片 |
| --- | ---: | --- | --- | --- |
| 新增无对白字幕视频，含人物介绍、台标和场景文字 | 6 | 6 段均为 `no_subtitle` | 6/6 一致 | 0 |
| 原有底部对白字幕视频 | 5 | 5 段均为 `with_subtitle` | 5/5 一致 | 5 个 `subtitle_cropped`，另对其中 1 段验证 `subtitle_blurred` |

结果只说明本批样本的行为符合预期，不代表所有字幕样式的识别准确率。

完整的原始视频和处理后视频不会提交到 Git 仓库；仓库仅包含上面的单帧对比图。
