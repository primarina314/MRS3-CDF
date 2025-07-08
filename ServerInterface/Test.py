import os
import cv2

from ImgToPkg_Interface import compress_img_mult_tgs_server

print("== 테스트 코드 진입 ==")
print("현재 작업 디렉토리:", os.getcwd())

img_path = 'test_image.jpg'  # 테스트용 이미지 파일 경로
output_path = 'test_output'  # 결과 저장 폴더

print("이미지 존재:", os.path.exists(img_path))

img = cv2.imread(img_path)
print("이미지 shape:", img.shape)

roi_point_lists = [
    [[0, 0], [10, 0], [10, 10], [0, 10]],    # 사각형 ROI 1개
    [[20, 20], [30, 20], [30, 30]],                # 삼각형 ROI 1개
]

for points in roi_point_lists:
    print("ROI 좌표:", points)
    for x, y in points:
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            print(f"경고: ({x},{y}) 좌표가 이미지 영역 밖에 있음!")

scaler = 2  # 2배 다운스케일

pkg_path = compress_img_mult_tgs_server(
    img_path=img_path,
    output_path=output_path,
    scaler=scaler,
    roi_point_lists=roi_point_lists,
    pkg_filename='output.pkg',       # 저장될 pkg 파일명
    interpolation=3
)

print(f"패키지 저장 완료! 위치: {pkg_path}")
