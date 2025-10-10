import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

def prepare_watermark(img_fft, watermark, alpha, private_key=None):
	"""
	准备水印数据并嵌入频谱
	:param img_fft: 图片的频谱数据
	:param watermark: 水印图片
	:param alpha: 水印强度
	:param private_key: 私钥，用于加密水印位置，如果为None则使用图像尺寸作为种子
	:return: 嵌入水印后的频谱数据
	"""
	height, width, channel = img_fft.shape
	wm_height, wm_width = watermark.shape[0], watermark.shape[1]
	x, y = list(range(height // 2)), list(range(width))
	seed = private_key if private_key is not None else (height + width)
	random.seed(seed)
	random.shuffle(x)
	random.shuffle(y)
	temp = np.zeros(img_fft.shape)
	for i in range(height // 2):
		for j in range(width):
			if x[i] < wm_height and y[j] < wm_width:
				temp[i][j] = watermark[x[i]][y[j]]
				temp[height - 1 - i][width - 1 - j] = temp[i][j]
	return img_fft + alpha * temp

def save_image(path, image):
	"""
	保存图片到指定路径
	:param path: 保存路径
	:param image: 图片数据
	"""
	dir_name = os.path.dirname(path)
	if dir_name and not os.path.exists(dir_name):
		os.makedirs(dir_name)
	cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def resize_watermark(watermark, ori_shape):
	"""
	调整水印图片大小，使其宽和高不超过原始图片的一半
	:param watermark: 水印图片
	:param ori_shape: 原始图片形状
	:return: 调整后的水印图片
	"""
	ori_height, ori_width = ori_shape[:2]
	wm_height, wm_width = watermark.shape[:2]
	while wm_height > ori_height or wm_width > ori_width:
		watermark = cv2.resize(watermark, (wm_width // 3, wm_height // 3), interpolation=cv2.INTER_AREA)
		wm_height, wm_width = watermark.shape[:2]
	return watermark

def encode(img_path, wm_path, output_path, private_key=None):
	"""
	给图片添加水印
	:param img_path: 输入图片路径
	:param wm_path: 水印图片路径
	:param output_path: 输出图片路径
	:param private_key: 私钥，用于加密水印
	"""
	alpha = 10
	if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
		output_path = output_path + '.png'
	img = cv2.imread(img_path)
	if img is None:
		return
	watermark = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
	if watermark is None:
		return
	
	# 检查并调整水印大小
	watermark = resize_watermark(watermark, img.shape)

	img_fft = np.fft.fft2(img)
	result_fft = prepare_watermark(img_fft, watermark, alpha, private_key)
	result = np.real(np.fft.ifft2(result_fft))
	save_image(output_path, result)
	print('[INFO]: Encode Successfully.')

def extract_watermark(watermark, shape, private_key=None):
	"""
	提取水印数据
	:param watermark: 水印频谱数据
	:param shape: 原始图片形状
	:param private_key: 私钥，用于解密水印位置，必须与嵌入时使用的私钥相同
	:return: 提取的水印图片
	"""
	height, width = shape[0], shape[1]
	result = np.zeros(watermark.shape)
	seed = private_key if private_key is not None else (height + width)
	random.seed(seed)
	x = list(range(height // 2))
	y = list(range(width))
	random.shuffle(x)
	random.shuffle(y)
	for i in range(height // 2):
		for j in range(width):
			result[x[i]][y[j]] = watermark[i][j]
	return result

def decode(img_path, origin_path, output_path, private_key=None):
	"""
	从图片中提取水印
	:param img_path: 输入图片路径
	:param origin_path: 原始图片路径
	:param output_path: 输出水印路径
	:param private_key: 私钥，用于解密水印，必须与嵌入时使用的私钥相同
	"""
	alpha = 10
	if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
		output_path = output_path + '.png'
	img = cv2.imread(img_path)
	if img is None:
		return
	origin_img = cv2.imread(origin_path)
	if origin_img is None:
		return
	origin_img_fft = np.fft.fft2(origin_img)
	img_fft = np.fft.fft2(img)
	watermark = (origin_img_fft - img_fft) / alpha
	watermark = np.real(watermark)
	result = extract_watermark(watermark, origin_img.shape, private_key)
	save_image(output_path, result)
	print('[INFO]: Decode Successfully.')

def tile_watermark(watermark, target_shape):
	"""
	将水印平铺到目标形状
	:param watermark: 原始水印图片
	:param target_shape: 目标形状 (height, width)
	:return: 平铺后的水印
	"""
	wm_height, wm_width = watermark.shape
	target_height, target_width = target_shape
	tiled_watermark = np.tile(watermark, (target_height // wm_height + 1, target_width // wm_width + 1))
	return tiled_watermark[:target_height, :target_width]

def restore_original(img_path, wm_path, output_path, private_key=None):
	"""
	从带水印的图片中恢复原始图片
	:param img_path: 带水印的图片路径
	:param wm_path: 水印图片路径
	:param output_path: 输出恢复的原始图片路径
	:param private_key: 私钥，必须与添加水印时使用的相同，否则会导致图像严重失真
	"""
	alpha = 10
	if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
		output_path = output_path + '.png'
	img = cv2.imread(img_path)
	if img is None:
		return
	watermark = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
	if watermark is None:
		return
	img_fft = np.fft.fft2(img)
	height, width, channel = img_fft.shape
	wm_height, wm_width = watermark.shape[0], watermark.shape[1]
	x, y = list(range(height // 2)), list(range(width))
	seed = private_key if private_key is not None else (height + width)
	random.seed(seed)
	random.shuffle(x)
	random.shuffle(y)
	temp = np.zeros(img_fft.shape)
	for i in range(height // 2):
		for j in range(width):
			if x[i] < wm_height and y[j] < wm_width:
				temp[i][j] = watermark[x[i]][y[j]]
				temp[height - 1 - i][width - 1 - j] = temp[i][j]
	original_fft = img_fft - alpha * temp
	original = np.clip(np.real(np.fft.ifft2(original_fft)), 0, 255).astype(np.uint8)
	save_image(output_path, original)
	print('[INFO]: Restore Successfully.')

def hash_private_key(key_str):
	"""
	将字符串私钥转换为数值
	:param key_str: 字符串私钥
	:return: 数值私钥
	"""
	if key_str is None:
		return None
	if isinstance(key_str, str):
		return sum(ord(c) * (i + 1) for i, c in enumerate(key_str))
	return key_str

def generate_text_watermark(text, font_path="asset/msyh.ttc", output_dir="images/wm", ori_shape=None):
	"""
	生成白底黑字的水印图片，并保存到指定目录
	:param text: 水印文本
	:param font_path: 字体文件路径，默认使用微软雅黑加粗字体
	:param output_dir: 水印图片保存目录
	:param ori_shape: 原始图片形状，用于检查水印大小
	:return: 水印图片路径
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	font_size = 50
	try:
		font = ImageFont.truetype(font_path, font_size)
	except OSError:
		font = ImageFont.load_default()

	if hasattr(font, "getbbox"):
		text_width, text_height = font.getbbox(text)[2:4]
	else:
		text_width, text_height = font.getsize(text)
	width, height = text_width + 20, text_height + 20
	
	image = Image.new("RGB", (width, height), "white")
	draw = ImageDraw.Draw(image)
	text_x = (width - text_width) // 2
	text_y = (height - text_height) // 2
	draw.text((text_x, text_y), text, fill="black", font=font)

	filename = f"{text}.png".replace(" ", "_")
	wm_path = os.path.join(output_dir, filename)
	watermark = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

	# 检查并调整水印大小
	if ori_shape is not None:
		watermark = resize_watermark(watermark, ori_shape)

	cv2.imwrite(wm_path, watermark)
	return wm_path