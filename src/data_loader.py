# downloader.py
import urllib.request
import os
from datetime import datetime, timedelta

def download_satellite_images(start_date_str, days, time_utc_str, save_dir):
    """
    지정된 기간 동안의 GK2A 적외선 위성 영상을 다운로드합니다.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
    except ValueError:
        print("오류: 날짜 형식이 잘못되었습니다. 'YYYY/MM/DD' 형식으로 입력해주세요.")
        return

    h, m = time_utc_str.split('.')
    total_files = 0
    error_count = 0

    print(f"{start_date.strftime('%Y%m%d')}부터 총 {days}일간의 {h}시 {m}분 적외선 영상을 다운로드합니다.")

    for i in range(days):
        current_date = start_date + timedelta(days=i)
        Y = current_date.strftime("%Y")
        M = current_date.strftime("%m")
        D = current_date.strftime("%d")

        URL = f"https://nmsc.kma.go.kr/IMG/GK2A/AMI/PRIMARY/L1B/COMPLETE/KO/{Y}{M}/{D}/{h}/gk2a_ami_le1b_ir105_ko020lc_{Y}{M}{D}{h}{m}.png"
        out_file = os.path.join(save_dir, f"ir_{Y}{M}{D}_{h}{m}.png")

        try:
            urllib.request.urlretrieve(URL, out_file)
            # 유효 파일 검사 (파일 크기가 10KB 미만이면 오류로 간주)
            if os.path.getsize(out_file) < 10000:
                print(f"오류: '{out_file}' 파일이 비정상적입니다. (크기 작음)")
                os.remove(out_file) # 불완전한 파일 삭제
                error_count += 1
            else:
                total_files += 1
        except Exception as e:
            # print(f"다운로드 실패: {URL} - {e}")
            error_count += 1
    
    print(f"총 {total_files}개의 자료 저장 완료 (오류 {error_count}회)")