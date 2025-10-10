# FFTSteg: FFT图片隐写术

FFTSteg 是一个基于FFT的图片隐写术工具，支持对图片进行水印嵌入、提取和恢复操作，支持使用私钥保护水印。

---

## 🌟 团队介绍
团队组成是来自华南理工大学（SCUT）电子与信息学院2023级信息工程（创新班）的4名本科学生（笔画排序）：

**马梓豪、孙艺、李昊峻、曾文博**

---

## 功能概览
- **编码（Encode）**: 将水印嵌入到图片中，可选图片水印或文字水印。
- **解码（Decode）**: 从带水印的图片中提取水印。
- **恢复（Restore）**: 从带水印的图片中恢复原始图片。
- **私钥保护**: 使用私钥加密水印和原图。

---

## 使用步骤

### Step 0: 环境配置
1. 下载并安装 [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)。
2. 使用YML文件创建并激活 `FFTSteg` 环境：
   ```bash
   conda env create -f asset/FFTSteg.yml
   conda activate FFTSteg
   ```

---

### Step 1: 编码（添加水印）

#### 参数说明
- `--mode`: 操作模式，设置为 `img_encode` 或 `txt_encode`。
  - `img_encode`: 使用图片作为水印。
  - `txt_encode`: 使用文本生成水印。
- `--ori_img`: 原始图片路径（输入）。
- `--wm_img`: 水印图片路径（输入，仅在 `img_encode` 模式下需要）。
- `--wm_text`: 自定义水印文本（仅在 `txt_encode` 模式下需要）。
- `--enc_img`: 编码后图片输出路径。
- `--private_key`: 私钥（可选），用于加密水印。
**当不输入`--wm_img`、`--wm_text`,将使用默认文本参数“FFT图片隐写术”**

#### 示例
1. 使用图片水印：
   ```bash
   python main.py \
       --mode img_encode \
       --ori_img images/ori/DuXing.png \
       --wm_img images/wm/SCUT.png \
       --enc_img images/enc/DuXing_SCUT_enc.png \
       --private_key siry
   ```
2. 使用文本水印：
   ```bash
   python main.py \
       --mode txt_encode \
       --ori_img images/ori/test1.png \
       --enc_img images/enc/test1_wm1_enc.png \
       --wm_text "生如夏花之绚烂" \
       --private_key siry
   ```
---

### Step 2: 解码（提取水印）

#### 参数说明
- `--mode`: 操作模式，设置为 `decode`。
- `--enc_img_decode`: 编码图片路径（输入）。
- `--ori_img_decode`: 原始图片路径（输入）。
- `--dwm_img`: 解码水印输出路径。
- `--private_key`: 私钥（可选），必须与编码时相同。

#### 示例
```bash
python main.py \
    --mode decode \
    --enc_img_decode images/enc/DuXing_SCUT_enc.png \
    --ori_img_decode images/ori/DuXing.png \
    --dwm_img images/dwm/DuXing_SCUT_dwm.png \
    --private_key siry
```

---

### Step 3: 恢复（从带水印图片恢复原始图片）

#### 参数说明
- `--mode`: 操作模式，设置为 `restore`。
- `--enc_img_restore`: 编码图片路径（输入）。
- `--wm_img_restore`: 水印图片路径（输入）。
- `--re_img`: 恢复图片输出路径。
- `--private_key`: 私钥（可选），必须与编码时相同。

#### 示例
```bash
python main.py \
    --mode restore \
    --enc_img_restore images/enc/DuXing_SCUT_enc.png \
    --wm_img_restore images/wm/SCUT.png \
    --re_img images/re/DuXing_SCUT_re.png \
    --private_key siry
```

---

## 私钥保护说明
1. **私钥作用**: 确定水印在频域中的嵌入位置，确保只有正确的私钥才能提取或恢复数据。
2. **私钥格式**: 任意字符串，系统会将其转换为数值用作随机种子。
3. **注意事项**:
   - 解码和恢复时必须使用与编码时相同的私钥。
   - 使用错误的私钥会导致水印提取失败或图像严重失真。
