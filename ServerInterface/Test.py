import os
import cv2
import numpy as np

# ---- 압축/복원 모듈 임포트 ----
from ImgToPkg_Interface import compress_img_mult_tgs_server
from PkgToImg_Interface import unpack_files, restore_img_mult_tgs

def main():
    # 1. 입력 이미지 준비
    img_path = 'test_image.jpg'
    output_dir = 'test_output'
    pkg_file = 'output.pkg'
    unpack_dir = 'unpacked'
    restored_dir = 'restored'
    scaler = 2  # 업스케일/다운스케일 배율 (예: 2배)
    interpolation = cv2.INTER_CUBIC  # 또는 cv2.INTER_LINEAR, INTER_AREA

    # 2. ROI 예시 (이미지 내 실제 좌표로 지정!)
    roi_point_lists = [
        [[0, 0], [10, 0], [10, 10], [0, 10]],    # 사각형 ROI 1개
        [[20, 20], [30, 20], [30, 30]],                # 삼각형 ROI 1개
    ]

    # 3. 압축(패키지) 생성
    print("=== 패키지 생성(압축) ===")
    pkg_path = compress_img_mult_tgs_server(
        img_path=img_path,
        output_path=output_dir,
        scaler=scaler,
        roi_point_lists=roi_point_lists,
        pkg_filename=pkg_file,
        interpolation=interpolation
    )
    print(f"PKG 파일 생성됨: {pkg_path}")

    # 4. 패키지 해제
    print("\n=== 패키지 해제 ===")
    unpack_files(pkg_path, unpack_dir)
    print(f"패키지 해제 완료: {unpack_dir}")

    # 5. 복원 이미지 생성
    print("\n=== 복원 이미지 생성 ===")
    restore_img_mult_tgs(
        input_path=unpack_dir,
        mrs3_mode=interpolation,
        output_path=restored_dir
    )

    # 결과 이미지 경로
    print(f"\n복원 결과 이미지는 {restored_dir}/restored.png 에 저장됩니다.")

if __name__ == '__main__':
    main()

