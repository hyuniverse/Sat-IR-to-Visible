# main.py
import argparse
import os
import gdown
from data_loader import download_satellite_images
from inference import translate_images

# [TODO] : 향후 train 모드 추가
def main():
    parser = argparse.ArgumentParser(description="적외선 위성 영상을 다운로드하고 가시광선 영상으로 변환합니다.")
    parser.add_argument("--start_date", type=str, required=True, help="다운로드 시작 날짜 (YYYY/MM/DD)")
    parser.add_argument("--days", type=int, required=True, help="다운로드할 기간 (일)")
    parser.add_argument("--time_utc", type=str, default="03.10", help="조회할 UTC 시간 (hh.mm)")
    parser.add_argument("--save_path", type=str, default="./data", help="데이터 저장 기본 경로")
    parser.add_argument("--model_path", type=str, default="./models/Pix2Pix_Generator_for_Facades.pt", help="학습된 생성자 모델 파일 경로")

    args = parser.parse_args()

    GDRIVE_FILE_ID = "1RtRLv7RYsCG_e9QZQHvQ_5hepCRqmNOC"
    model_path = args.model_path

    # 0. 모델 파일이 없으면 다운로드
    if not os.path.exists(model_path):
        print(f"'{model_path}'를 찾을 수 없습니다. 구글 드라이브에서 모델 파일을 다운로드합니다...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            gdown.download(id=GDRIVE_FILE_ID, output=model_path, quiet=False)
            print("모델 다운로드가 완료되었습니다.")
        except Exception as e:
            print(f"오류: 모델 파일 다운로드에 실패했습니다. {e}")
            return # 다운로드 실패 시 프로그램 종료

    print("--- 추론 모드를 시작합니다. ---")

    # 1. 위성 영상 다운로드
    print("--- 1. 위성 영상 다운로드를 시작합니다. ---")
    ir_image_dir = os.path.join(args.save_path, "infrared")
    os.makedirs(ir_image_dir, exist_ok=True)
    
    download_satellite_images(
        start_date_str=args.start_date,
        days=args.days,
        time_utc_str=args.time_utc,
        save_dir=ir_image_dir
    )
    print(f"--- 다운로드 완료! 이미지가 '{ir_image_dir}' 폴더에 저장되었습니다. ---\n")

    # 2. 영상 변환
    print("--- 2. 적외선 -> 가시광선 영상 변환을 시작합니다. ---")
    output_image_dir = os.path.join(args.save_path, "visible_generated")
    os.makedirs(output_image_dir, exist_ok=True)
    
    translate_images(
        input_dir=ir_image_dir,
        output_dir=output_image_dir,
        model_path=args.model_path
    )
    print(f"--- 변환 완료! 결과가 '{output_image_dir}' 폴더에 저장되었습니다. ---")

if __name__ == "__main__":
    main()