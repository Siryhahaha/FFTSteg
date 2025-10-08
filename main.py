import argparse
from utils import encode, decode, restore_original, hash_private_key, compress_image, generate_text_watermark
import cv2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, required=True, choices=['encode', 'decode', 'restore', 'compress'])

	parser.add_argument('--ori_img', type=str)
	parser.add_argument('--wm_img', type=str)
	parser.add_argument('--enc_img', type=str)

	parser.add_argument('--enc_img_decode', type=str)
	parser.add_argument('--ori_img_decode', type=str)
	parser.add_argument('--dwm_img', type=str)

	parser.add_argument('--enc_img_restore', type=str)
	parser.add_argument('--wm_img_restore', type=str)
	parser.add_argument('--re_img', type=str)

	parser.add_argument('--input_img_compress', type=str)
	parser.add_argument('--output_img_compress', type=str)
	parser.add_argument('--quality', type=int, default=85)
	
	parser.add_argument('--private_key', type=str)
	args = parser.parse_args()

	private_key = hash_private_key(args.private_key) if args.private_key else None
	
	if args.mode == 'encode':
		if not args.wm_img:
			img = cv2.imread(args.ori_img)
			height, width, _ = img.shape
			watermark = generate_text_watermark("FFT图片隐写术", (width // 2, height // 2))
			wm_path = "generated_watermark.png"
			cv2.imwrite(wm_path, watermark)
			args.wm_img = wm_path
		encode(img_path=args.ori_img, wm_path=args.wm_img, output_path=args.enc_img, private_key=private_key)
	
	elif args.mode == 'decode':
		decode(img_path=args.enc_img_decode, origin_path=args.ori_img_decode, output_path=args.dwm_img, private_key=private_key)
	
	elif args.mode == 'restore':
		restore_original(img_path=args.enc_img_restore, wm_path=args.wm_img_restore, output_path=args.re_img, private_key=private_key)
	
	elif args.mode == 'compress':
		compress_image(input_path=args.input_img_compress, output_path=args.output_img_compress, quality=args.quality)

if __name__ == '__main__':
	main()
