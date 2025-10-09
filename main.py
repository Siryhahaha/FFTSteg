import argparse
from utils import encode, decode, restore_original, hash_private_key, generate_text_watermark
import cv2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, required=True, choices=['img_encode', 'txt_encode', 'decode', 'restore'])

	parser.add_argument('--ori_img', type=str)
	parser.add_argument('--wm_img', type=str)
	parser.add_argument('--enc_img', type=str)

	parser.add_argument('--enc_img_decode', type=str)
	parser.add_argument('--ori_img_decode', type=str)
	parser.add_argument('--dwm_img', type=str)

	parser.add_argument('--enc_img_restore', type=str)
	parser.add_argument('--wm_img_restore', type=str)
	parser.add_argument('--re_img', type=str)

	parser.add_argument('--private_key', type=str)
	parser.add_argument('--wm_text', type=str, default="FFT图片隐写术", help="自定义水印文本")
	args = parser.parse_args()

	private_key = hash_private_key(args.private_key) if args.private_key else None
	
	if args.mode == 'img_encode':
		encode(img_path=args.ori_img, wm_path=args.wm_img, output_path=args.enc_img, private_key=private_key)

	elif args.mode == 'txt_encode':
		wm_path = generate_text_watermark(args.wm_text)
		encode(img_path=args.ori_img, wm_path=wm_path, output_path=args.enc_img, private_key=private_key)

	elif args.mode == 'decode':
		decode(img_path=args.enc_img_decode, origin_path=args.ori_img_decode, output_path=args.dwm_img, private_key=private_key)
	
	elif args.mode == 'restore':
		restore_original(img_path=args.enc_img_restore, wm_path=args.wm_img_restore, output_path=args.re_img, private_key=private_key)

if __name__ == '__main__':
	main()
