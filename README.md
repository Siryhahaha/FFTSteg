# FFTSteg: FFT图片隐写术

FFTSteg 是一个基于傅里叶变换（FFT）的图片隐写术工具，支持对图片进行水印嵌入、提取和恢复操作。支持使用私钥保护水印，确保只有正确的私钥才能提取或恢复数据。

---

## 🌟 团队介绍
团队组成是来自华南理工大学（SCUT）电子与信息学院2023级信息工程（创新班）的4名本科学生（笔画排序）：

**马梓豪、李昊峻、孙艺、曾文博**

---

## 📋 功能概览
- **编码（Encode）**: 将水印嵌入到图片中。
- **解码（Decode）**: 从带水印的图片中提取水印。
- **恢复（Restore）**: 从带水印的图片中恢复原始图片。
- **私钥保护**: 使用私钥加密水印，确保只有正确的私钥才能提取或恢复数据。

---

## 🛠️ 使用步骤

### Step 0: 环境配置
1. 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。
2. 使用YML文件创建并激活 `FFTSteg` 环境：
   ```bash
   conda env create -f FFTSteg.yml
   conda activate FFTSteg
   ```

---

### Step 1: 编码（添加水印）

#### 参数说明
- `--mode`: 操作模式，设置为 `encode`。
- `--ori_img`: 原始图片路径（输入）。
- `--wm_img`: 水印图片路径（输入，可选）。
- `--enc_img`: 编码后图片输出路径。
- `--private_key`: 私钥（可选），用于加密水印。

#### 标准格式
```bash
python main.py \
    --mode encode \
    --ori_img <原始图片路径> \
    --enc_img <编码图片输出路径> \
    [--wm_img <水印图片路径>] \
    [--private_key <私钥>]
```

#### 示例
1. 使用自定义水印图片：
   ```bash
   python main.py \
       --mode encode \
       --ori_img images/ori/test1.png \
       --wm_img images/wm/wm1.png \
       --enc_img images/enc/test1_wm1_enc.png \
       --private_key siry
   ```
2. 自动生成默认文本水印：
   ```bash
   python main.py \
       --mode encode \
       --ori_img images/ori/test1.png \
       --enc_img images/enc/test1_wm1_enc.png \
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

#### 标准格式
```bash
python main.py \
    --mode decode \
    --enc_img_decode <编码图片路径> \
    --ori_img_decode <原始图片路径> \
    --dwm_img <解码水印输出路径> \
    [--private_key <私钥>]
```

#### 示例
```bash
python main.py \
    --mode decode \
    --enc_img_decode images/enc/test1_wm1_enc.png \
    --ori_img_decode images/ori/test1.png \
    --dwm_img images/dwm/test1_wm1_dwm.png \
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

#### 标准格式
```bash
python main.py \
    --mode restore \
    --enc_img_restore <编码图片路径> \
    --wm_img_restore <水印图片路径> \
    --re_img <恢复图片输出路径> \
    [--private_key <私钥>]
```

#### 示例
```bash
python main.py \
    --mode restore \
    --enc_img_restore images/enc/test1_wm1_enc.png \
    --wm_img_restore images/wm/wm1.png \
    --re_img images/re/test1_wm1_re.png \
    --private_key siry
```

---

## 🔑 私钥保护说明
1. **私钥作用**: 确定水印在频域中的嵌入位置，确保只有正确的私钥才能提取或恢复数据。
2. **私钥格式**: 任意字符串，系统会将其转换为数值用作随机种子。
3. **注意事项**:
   - 解码和恢复时必须使用与编码时相同的私钥。
   - 使用错误的私钥会导致水印提取失败或图像严重失真。

---

## 📌 注意事项
1. 水印强度已固定为10，无需手动设置。
2. 水印图片的尺寸应小于原始图片。
3. 输出路径可以不包含文件扩展名，系统会自动添加 `.png` 扩展名。
4. 恢复功能不再需要指定水印强度，系统会自动处理。
5. 使用错误的私钥进行恢复会导致图像严重失真，这是系统的安全特性。