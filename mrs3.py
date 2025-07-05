import numpy as np
import cv2
import time
import os
from collections import defaultdict
from PIL import Image
import configparser
import interpolation as inter
import gc
import utils


ROI_RECTANGLE = 0
ROI_POLYGON = 1

EDSR = -1
INTER_NEAREST = cv2.INTER_NEAREST
INTER_LINEAR = cv2.INTER_LINEAR
INTER_CUBIC = cv2.INTER_CUBIC
INTER_AREA = cv2.INTER_AREA
INTER_LANCZOS4 = cv2.INTER_LANCZOS4
INTER_LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
INTER_NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
INTER_MAX = cv2.INTER_MAX
WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP
WARP_RELATIVE_MAP = cv2.WARP_RELATIVE_MAP

drawing = False
points = []

t1 = 0
t2 = 0
t3 = 0

def _select_rectangle_roi(image_path):
    """
    input_path: 추출할 이미지 경로
    return: roi-ndarray, (from_y, to_y, from_x, to_x)
    """

    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return None, None

    # ROI(관심영역) 선택 창 띄우기 (마우스로 드래그)
    x, y, w, h = cv2.selectROI("select part to remain by dragging", img, showCrosshair=True, fromCenter=False)

    # ROI가 정상적으로 선택된 경우에만 진행
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        cv2.imshow("selected part", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return roi, (y, y+h, x, x+w)
    else:
        print("None of part selected")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("none")
        return None, None

def _draw_polygon(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
        drawing = False  # 다각형 닫기

def _select_polygon_roi(image_path):
    global drawing, points
    img = cv2.imread(image_path)
    clone = img.copy()
    cv2.namedWindow("indicate polygon in img")
    cv2.setMouseCallback("indicate polygon in img", _draw_polygon)

    drawing = True
    while drawing:
        temp = clone.copy()
        if len(points) > 0:
            cv2.polylines(temp, [np.array(points)], False, (0,255,0), 2)
            for pt in points:
                cv2.circle(temp, pt, 3, (0,0,255), -1)
        cv2.imshow("indicate polygon in img", temp)
        key = cv2.waitKey(1)
        if key == 27:  # ESC로 취소
            points = []
            break
        if key == ord('s') and len(points) >= 3:  # 's'로 저장
            drawing = False

    if len(points) >= 3:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyWindow('mask')

        roi = cv2.bitwise_and(img, img, mask=mask)
        # 다각형의 bounding box로 crop
        pts = np.array(points)
        x, y, w, h = cv2.boundingRect(pts)
        cropped = roi[y:y+h, x:x+w]
        cv2.imshow("polygon ROI", cropped)
        cv2.waitKey(0)
        cropped_mask = mask[y:y+h, x:x+w]
        cv2.imshow('cropped mask', cropped_mask)
        cv2.waitKey(0)
        print(f"cropped target: {cropped.shape}")
        print(f"cropped mask: {cropped_mask.shape}")
        
    else:
        print("3개 이상의 꼭짓점이 필요합니다.")
        return None, None

    drawing = False
    points = []
    cv2.destroyAllWindows()
    return cropped, (y, y+h, x, x+w)

# 사용 예시
# cv2.imshow('cropped', select_polygon_roi('Lenna_(test_image).png'))


################################
# upscale
# TODO: 메모리 한계 넘어가는 큰 이미지는 분할해서. 분할된 경계가 조금씩 겹치도록 한 후, 여기에도 자연스럽게 blending
################################

def _upscale_by_edsr(image_path, scaler):
    t1 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    if scaler not in [2, 3, 4]:
        print(f"Invalid scaler value: {scaler}. Must be 2, 3 or 4.")
        return None
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-enabled GPU found.")
        return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(f'models/EDSR_x{scaler}.pb')
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

    # gpu acceleration
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    sr.setModel('edsr', scaler)

    try:
        result = sr.upsample(img)
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return None
    t2 = time.time()
    print(f'{t2-t1} sec taken')
    return result

def _upscale_by_resize(image_path, scaler, interpolation = cv2.INTER_CUBIC):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    h, w = img.shape[0] * scaler, img.shape[1] * scaler
    result = cv2.resize(img, (w, h), interpolation=interpolation)
    return result

################################
# downscale
################################
def _downscale_img(image_path, scaler, interpolation = cv2.INTER_AREA):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    new_h, new_w = img.shape[0]//scaler, img.shape[1]//scaler

    result = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return result


def _combine_images(A, B):
    """
    검은색 픽셀을 기준으로 두 넘파이배열(이미지를 합성)
    A: (H, W, 3) shape의 넘파이 배열 (검은색 픽셀 기준)
    B: (H, W, 3) shape의 넘파이 배열 (A의 검은 픽셀 대체용)
    """
    # 1. A 이미지에서 검은색 픽셀 마스크 생성
    black_mask = np.all(A == [0, 0, 0], axis=2)
    
    # 2. 3채널에 적용 가능하도록 차원 확장
    mask_3d = black_mask[:, :, np.newaxis]
    
    # 3. 조건에 따라 픽셀 선택
    return np.where(mask_3d, B, A)


def _rename_images_by_resolution(folder_path):
    """
    폴더 내의 모든 이미지 파일명을 모두 '해상도.png' 로 변경
    e.g. 1920x1080.png
    """

    # 해상도별로 파일 개수를 기록할 딕셔너리
    resolution_count = defaultdict(int)

    # 폴더 내 모든 png 파일 목록
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                resolution = f"{width}x{height}"

                # 해당 해상도 파일 개수 증가
                resolution_count[resolution] += 1
                count = resolution_count[resolution]

                # 새 파일명 생성
                if count == 1:
                    new_name = f"{resolution}.png"
                else:
                    new_name = f"{resolution}-{count-1}.png"

                new_path = os.path.join(folder_path, new_name)

                # 파일명 변경
                os.rename(file_path, new_path)
                print(f"Renamed '{file_name}' to '{new_name}'")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 사용 예시 (폴더 경로를 원하는 경로로 바꿔서 사용)
# rename_images_by_resolution('sample-images-png')


#########################
# 이미지 내의 검은색(0 0 0) 픽셀 비율
#########################

def _black_pixel_ratio(image_path):
    """
    이미지 내의 0 0 0 픽셀 비율을 리턴
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    # (H, W, 3) 배열에서 [0,0,0]인 픽셀 찾기
    black_mask = np.all(img == [0, 0, 0], axis=2)
    # axis=2: channel -> (h, w, c) 에서 c 가 사라진 (h, w) 로 리턴 shape
    black_count = np.sum(black_mask)
    total_pixels = img.shape[0] * img.shape[1]
    ratio = black_count / total_pixels

    print(f"검은 픽셀 개수: {black_count}")
    print(f"전체 픽셀 개수: {total_pixels}")
    print(f"검은 픽셀 비율: {ratio:.4%}")
    return ratio



# mrs3 적용 후 파일저장 경로/이름
roi_filename = 'roi' # png
downscaled_filename = 'downscaled' # png
config_filename = 'config' # ini
restored_filename = 'restored' # png
roi_binary_filename = 'bin' # png

# mrs3 mode, select roi mode
def compress_img(img_path, output_path, scaler, roi_mode, interpolation=INTER_AREA):
    """
    단일 타겟 대상 압축
    img_path: mrs3 적용할 이미지 경로
    output_path: 결과 저장할 폴더 경로
    scaler: 이미지 downscale 배율
    """

    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    # TODO: 타겟 개수 조정할 수 있도록 추가

    # 이미지 축소 및 roi 저장
    if roi_mode == ROI_RECTANGLE:
        roi, loc = _select_rectangle_roi(img_path)
    
    if roi_mode == ROI_POLYGON:
        roi, loc = _select_polygon_roi(img_path)

    # TODO: 다양한 interpolation 비교 및 복원 비교
    downscaled = _downscale_img(img_path, scaler, interpolation=interpolation)


    # 메타데이터 ini 에 저장
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'SCALER': f'{scaler}',
        'NUMBER_OF_TARGETS': f'1'
    }
    config['0'] = {
        'Y_FROM': f'{loc[0]}',
        'Y_TO': f'{loc[1]}',
        'X_FROM': f'{loc[2]}',
        'X_TO': f'{loc[3]}'
    }
    # TODO: 타겟 넘버링하여 각자 위치 정보 저장

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # TODO: 타겟 이미지 저장 경로 리스트로 0부터 n-1 까지 순차저장
    cv2.imwrite(f'{output_path}/{downscaled_filename}.png', downscaled)
    cv2.imwrite(f'{output_path}/{roi_filename}{0}.png', roi)
    with open(f'{output_path}/{config_filename}.ini', 'w') as configfile:
        config.write(configfile)

    filesize_bef = os.path.getsize(img_path)
    filesize_downscaled = os.path.getsize(f'{output_path}/{downscaled_filename}.png')
    filesize_roi = os.path.getsize(f'{output_path}/{roi_filename}{0}.png')
    filesize_config = os.path.getsize(f'{output_path}/{config_filename}.ini')

    print(f'original file: {filesize_bef}')
    print(f'downscaled filesize: {filesize_downscaled}')
    print(f'roi filesize: {filesize_roi}')
    print(f'config filesize: {filesize_config}')

    # 파일 사이즈 압축률 print
    print(f'compression ratio: {(filesize_downscaled + filesize_roi + filesize_config) / filesize_bef}')

    return

def restore_img(input_path, mrs3_mode, output_path=""):
    """
    mrs3 처리한 후, 이미지 복원
    input_path: mrs3 적용한 폴더 경로 - 나중에 폴더말고 하나의 파일형식에 저장하도록 수정하는게 좋을듯.
    """

    if not os.path.exists(f'{input_path}/{downscaled_filename}.png'):
        print(f"Error loading image: {input_path}/{downscaled_filename}.png")
        return

    if not os.path.exists(f'{input_path}/{config_filename}.ini'):
        print(f'Error loading config: {input_path}/{config_filename}.ini')
        return
    
    if not os.path.exists(f'{input_path}/{roi_filename}{0}.png'):
        print(f'Error loading image: {input_path}/{roi_filename}{0}.png')
        return

    config = configparser.ConfigParser()
    config.read(f'{input_path}/{config_filename}.ini')


    y_from, y_to, x_from, x_to = int(config['0']['Y_FROM']), int(config['0']['Y_TO']), int(config['0']['X_FROM']), int(config['0']['X_TO'])
    scaler = int(config['DEFAULT']['SCALER'])

    if mrs3_mode == EDSR:
        restored = _upscale_by_edsr(f'{input_path}/{downscaled_filename}.png', scaler=scaler)
    else:
        restored = _upscale_by_resize('{input_path}/{downscaled_path}.png', scaler=scaler, interpolation=mrs3_mode)
    

    # TODO: 다수 roi 반영 0 ~ n-1
    roi = cv2.imread(f'{input_path}/{roi_filename}{0}.png')    
    combined_roi = _combine_images(roi, restored[y_from:y_to, x_from:x_to])
    restored[y_from:y_to, x_from:x_to] = combined_roi

    cv2.imshow('restored img', restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path == "":
        cv2.imwrite(f'{restored_filename}.png', restored)
        return

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(f'{output_path}/{restored_filename}.png', restored)
    return


################ 위까지는 블렌딩 없이 구현 ###################



# 블렌딩 시에 고려할 점
# 여러타켓
# 타겟이 주어지면 해당 타겟 주위로 블렌딩 시행

# 곡선 경계로의 확장을 고려해서 클릭한 점들의 위치좌표를 넘겨주는 방식이 아니라, findContours 사용


def _point_line_distance(point, line_start, line_end):
    """
    점과 선분 사이의 최소 수직거리 계산
    point: (x, y) 튜플
    line_start, line_end: 선분의 두 끝점 (x, y) 튜플
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 선분 벡터
    line_vec = np.array([x2 - x1, y2 - y1])
    # 점과 선분 시작점 벡터
    point_vec = np.array([px - x1, py - y1])

    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        # 선분이 점인 경우
        return np.linalg.norm(point_vec)

    # 점에서 선분으로의 투영 비율
    t = np.dot(point_vec, line_vec) / line_len_sq

    if t < 0.0:
        # 투영점이 선분 시작점 밖에 있는 경우
        closest_point = np.array([x1, y1])
    elif t > 1.0:
        # 투영점이 선분 끝점 밖에 있는 경우
        closest_point = np.array([x2, y2])
    else:
        # 투영점이 선분 위에 있는 경우
        closest_point = np.array([x1, y1]) + t * line_vec

    # 점과 가장 가까운 점 사이 거리
    dist = np.linalg.norm(np.array([px, py]) - closest_point)
    return dist


def _is_point_inside_polygon(point, vertices):
    """
    점이 다각형 내부에 있는지 확인 (Ray casting algorithm)
    point: (x, y) 튜플
    vertices: 다각형 꼭짓점 좌표 배열 [(x1,y1), (x2,y2), ...]
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside





def _distance_to_polygon_edge_from_mask(mask, point):
    """
    mask: 2D numpy array, 다각형 영역이 1(또는 255), 배경이 0인 바이너리 마스크
    point: (x, y) 튜플, 기준점 좌표

    반환값: point에서 다각형 경계까지의 최소 수직거리 (float)
    """
    # 마스크를 바이너리(0 또는 255)로 변환
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # 다각형 외곽선(경계) 검출
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None  # 다각형이 없는 경우
    
    # 다각형 꼭짓점 좌표 추출
    polygon = contours[0].reshape(-1, 2)

    # 점이 다각형 내부에 있는지 확인
    inside = _is_point_inside_polygon(point, polygon)

    min_dist = float('inf')
    for contour in contours:
        pts = contour.reshape(-1, 2)
        n = len(pts)
        for i in range(n):
            start = tuple(pts[i])
            end = tuple(pts[(i + 1) % n])
            dist = _point_line_distance(point, start, end)
            if dist < min_dist:
                min_dist = dist

    return min_dist if inside else -min_dist

def _distance_to_polygon_edge_from_contours(contours, point):
    """
    point: (x, y) 튜플, 기준점 좌표
    contours: contours 튜플(vertices). cv2.findContours 의 0번째 컴포넌트
    반환값: point에서 다각형 경계까지의 최소 수직거리 (float) - 부호 고려
    """

    if len(contours) == 0:
        return None  # 다각형이 없는 경우
    
    # 다각형 꼭짓점 좌표 추출
    polygon = contours[0].reshape(-1, 2)

    # 점이 다각형 내부에 있는지 확인
    inside = _is_point_inside_polygon(point, polygon)

    min_dist = float('inf')
    for contour in contours:
        pts = contour.reshape(-1, 2)
        n = len(pts)
        for i in range(n):
            start = tuple(pts[i])
            end = tuple(pts[(i + 1) % n])
            dist = _point_line_distance(point, start, end)
            if dist < min_dist:
                min_dist = dist
    return min_dist if inside else -min_dist

def _distance_to_polygon_edge_temp(vertices, point):
    global t1, t2, t3
    """
    polygon: contour 꼭짓점 ndarray. Shape 은 (꼭짓점 개수, 2)
    point: (x, y) 튜플
    """
    # 점이 다각형 내부에 있는지 확인
    t_i = time.time()
    inside = _is_point_inside_polygon(point, vertices)
    t_f = time.time()
    t1 += t_f - t_i

    t_i = time.time()
    min_dist = float('inf')
    n = len(vertices)
    for i in range(n):
        start = tuple(vertices[i])
        end = tuple(vertices[(i + 1) % n])
        dist = _point_line_distance(point, start, end)
        if dist < min_dist:
            min_dist = dist
    t_f = time.time()
    t2 += t_f - t_i
    return min_dist if inside else -min_dist

"""
필요한거


대안
binary 이미지도 저장
ㄴ전체에 대한 바이너리가 아니라, 각각에 대해 바이너리 지정
ㄴambiguity 제거를 위함
ㄴ용량 오버헤드는 원본부분 roi.png 의 1퍼센트 내외

2721, 33064, 473831
cv2.imwrite('bin-img.png', mask_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1])

압축 전략 지정
IMWRITE_PNG_STRATEGY
cv2.imwrite('output_image.png', image, [cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT])

cv2.IMWRITE_PNG_STRATEGY_DEFAULT (0): 기본 전략
cv2.IMWRITE_PNG_STRATEGY_FILTERED (1): 필터링 전략
cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY (2): 허프만 인코딩만 사용
cv2.IMWRITE_PNG_STRATEGY_RLE (3): RLE(Run-Length Encoding) 전략 (기본값)
cv2.IMWRITE_PNG_STRATEGY_FIXED (4): 고정 전략

cv2.imwrite('output_image.png', image, 
            [cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE,
             cv2.IMWRITE_PNG_COMPRESSION, 9])

TODO: 여러 압축전략 중 최적으로 결정하도록 설정
https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html

"""

# np.savez_compressed()

"""
TODO: 다수 타겟 지정(다각형) 및 바이너리 이미지로 저장

"""

# 하나의 점 집합/리스트에 대해 수행 중인지, 아니면 클릭시 새로운 집합을 만들어야 하는지에 대한 여부를 담은 bool 변수 필요
# 또는 완료된 점 집합 개수에 대한 변수 -> points 리스트의 길이와 비교해서 새로 만들지, 그대로 진행할지
multiple_points = []

roi_contours = []
roi_contour_num = 0

# len(contours) == contour_num -> 새로 추가
# len(contours) > contour_num -> 마지막 요소(contour_num번쨰 index)에 그대로 추가

def _draw_multiple_polygon(event, x, y, flags, param):
    global roi_contours, roi_contour_num
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_contours) == roi_contour_num:
            roi_contours.append([])
        roi_contours[roi_contour_num].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and roi_contours and len(roi_contours)!=roi_contour_num and len(roi_contours[-1]) >= 3:
        roi_contour_num += 1


def _select_multiple_polygon_roi(image_path):

    """
    타겟 다중 선택
    우클릭으로 다음 타겟으로 넘어가고, s키 눌러서 최종 저장
    리턴: (타겟 원본 부분 ndarray, 타겟 바이너리 mask, (y, y+h, x, x+w)) 튜플
    """
    global drawing, roi_contours, roi_contour_num
    img = cv2.imread(image_path)
    clone = img.copy()
    cv2.namedWindow("indicate polygon in img")
    cv2.setMouseCallback("indicate polygon in img", _draw_multiple_polygon)

    drawing = True
    roi_contours = []
    roi_contour_num = 0
    while drawing:
        temp = clone.copy()

        for i in range(len(roi_contours)):
            points = roi_contours[i]
            if len(points) > 0:
                cv2.polylines(temp, [np.array(points)], i < roi_contour_num, (0,255,0), 2)
                for pt in points:
                    cv2.circle(temp, pt, 3, (0,0,255), -1)
        cv2.imshow("indicate polygon in img", temp)

        key = cv2.waitKey(1)
        if key == 27:  # ESC로 취소
            roi_contours = []
            drawing = False
            break
        if key == ord('s') and roi_contours and len(roi_contours[-1]) >= 3:  # 's'로 저장
            drawing = False

    result = []
    for i in range(len(roi_contours)):
        points = roi_contours[i]

        if len(points) >= 3:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            roi = cv2.bitwise_and(img, img, mask=mask)
            # 다각형의 bounding box로 crop
            pts = np.array(points)
            x, y, w, h = cv2.boundingRect(pts)
            cropped = roi[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]

            result.append((cropped, cropped_mask, (y, y+h, x, x+w)))
            cv2.imshow(f"polygon ROI {i}", cropped)
            cv2.waitKey(0)
            cv2.destroyWindow(f"polygon ROI {i}")
        else:
            print("3개 이상의 꼭짓점이 필요합니다.")
            return None

    drawing = False
    roi_contours = []
    cv2.destroyAllWindows()
    return result


# mrs3 mode, select roi mode
"""
for multiple targets
TODO: 현재는 poly 만 multiple targets -> 이후에 rect 및 curves 도 도입
"""
def compress_img_mult_tgs(img_path, output_path, scaler, roi_mode, interpolation=INTER_AREA):
    """
    img_path: mrs3 적용할 이미지 경로
    output_path: 결과 저장할 폴더 경로
    scaler: 이미지 downscaling 배율
    roi_mode: 타겟 설정 방식(직사각형, 다각형, 곡선)
    interpolation: downscale 시에 사용할 interpolation
    """

    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return
    
    targets = _select_multiple_polygon_roi(img_path)

    # TODO: 다양한 interpolation 비교 및 복원 비교
    downscaled = _downscale_img(img_path, scaler, interpolation=interpolation)

    # 메타데이터 ini 에 저장
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'SCALER': f'{scaler}',
        'NUMBER_OF_TARGETS': f'{len(targets)}'
    }

    # 각 타겟 위치정보 저장
    for i in range(len(targets)):
        config[f'{i}'] = {
            'Y_FROM': f'{targets[i][2][0]}',
            'Y_TO': f'{targets[i][2][1]}',
            'X_FROM': f'{targets[i][2][2]}',
            'X_TO': f'{targets[i][2][3]}'
        }

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 축소 downscaled 이미지 저장
    cv2.imwrite(f'{output_path}/{downscaled_filename}.png', downscaled, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # 타겟 하나씩 순차저장(원본 부분 이미지, 영역 바이너리 이미지)
    for i in range(len(targets)):
        cv2.imwrite(f'{output_path}/{roi_filename}{i}.png', targets[i][0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(f'{output_path}/{roi_binary_filename}{i}.png', targets[i][1], [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1])

    with open(f'{output_path}/{config_filename}.ini', 'w') as configfile:
        config.write(configfile)

    # 압축률 계산
    filesize_bef = os.path.getsize(img_path)
    filesize_downscaled = os.path.getsize(f'{output_path}/{downscaled_filename}.png')
    filesize_roi = 0
    filesize_roi_bin = 0
    for i in range(len(targets)):
        filesize_roi += os.path.getsize(f'{output_path}/{roi_filename}{i}.png')
        filesize_roi_bin += os.path.getsize(f'{output_path}/{roi_binary_filename}{i}.png')
    filesize_config = os.path.getsize(f'{output_path}/{config_filename}.ini')

    print(f'original filesize: {filesize_bef}')
    print(f'downscaled filesize: {filesize_downscaled}')
    print(f'roi filesize: {filesize_roi}')
    print(f'roi bin filesize: {filesize_roi_bin}')
    print(f'config filesize: {filesize_config}')
    print(f'total filesize: {(filesize_downscaled + filesize_roi + filesize_roi_bin + filesize_config)}')

    # 파일 사이즈 압축률 print
    print(f'compression ratio: {(filesize_downscaled + filesize_roi + filesize_roi_bin + filesize_config) / filesize_bef}')
    
    return

BLEND_LINEAR = 0
BLEND_HERMIT_3 = 1
BLEND_HERMIT_5 = 2
BLEND_SINUSOIDAL = 3
BLEND_STEP = 4

_DIST_CRIT_COEF = .4

def _blend_images_with_contour_distance(A, B, contour, blend=BLEND_SINUSOIDAL):
    """
    A, B: 두 이미지 (같은 크기, 3채널, uint8)
    contour: 다각형 외곽선 (Nx1x2 형태의 numpy 배열)
    blend: 알파블렌딩 가중치 보간방식
    
    contour 외부는 A로 채우고, contour 내부는 contour로부터의 거리 기반 알파 블렌딩
    """
    # 이미지 크기
    h, w = A.shape[:2]
    
    # 빈 마스크 생성 (contour 내부 255, 외부 0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    
    # contour 내부 픽셀에 대해 거리 변환 (contour 경계까지 거리)
    dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
    # 이미지 크기에 따라 max_distance(블렌딩 두께) 다르게 설정
    max_distance = np.max(dist_transform) * _DIST_CRIT_COEF
    # 거리 임계값 이상인 부분은 B만 사용
    # 거리 임계값 이하인 부분은 거리 비례 가중치 계산
    
    # 거리 최대값으로 정규화 (0~1)
    if blend == BLEND_LINEAR:
        alpha = inter.np_linear(1 - dist_transform / max_distance, 0, 1)
    elif blend == BLEND_HERMIT_3:
        alpha = inter.np_hermit_3(1 - dist_transform / max_distance, 0, 1)
    elif blend == BLEND_HERMIT_5:
        alpha = inter.np_hermit_5(1 - dist_transform / max_distance, 0, 1)
    elif blend == BLEND_SINUSOIDAL:
        alpha = inter.np_sinusoidal(1 - dist_transform / max_distance, 0, 1)
    elif blend == BLEND_STEP:
        alpha = inter.np_unit_step(1 - dist_transform / max_distance, 0, 1)
    else:
        print("No such blend option")
        return None
    
    # contour 외부는 alpha=1 (A만 사용)
    alpha[mask == 0] = 1
    
    # alpha는 A의 가중치, (1-alpha)는 B의 가중치
    alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    # float32로 변환하여 가중치 곱셈
    A_f = A.astype(np.float32)
    B_f = B.astype(np.float32)
    
    blended = A_f * alpha_3ch + B_f * (1 - alpha_3ch)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

# TODO: 큰 이미지에 대해서는 upscale_large_img 로 처리 - 아니면 그냥 바로 upscale_large_img 로 하고 내부에 크기 판정 조건문 추가하는 식으로 처리
def restore_img_mult_tgs(input_path, mrs3_mode, output_path=""):
    global t1, t2, t3
    t1 = t2 = t3 = 0
    """
    mrs3 처리한 후, 이미지 복원
    input_path: mrs3 적용한 폴더 경로 - 나중에 폴더말고 하나의 파일형식에 저장하도록 수정하는게 좋을듯.
    """

    if not os.path.exists(f'{input_path}/{downscaled_filename}.png'):
        print(f"Error loading image: {input_path}/{downscaled_filename}.png")
        return

    if not os.path.exists(f'{input_path}/{config_filename}.ini'):
        print(f'Error loading config: {input_path}/{config_filename}.ini')
        return
    
    if not os.path.exists(f'{input_path}/{roi_filename}{0}.png'):
        print(f'Error loading image: {input_path}/{roi_filename}{0}.png')
        return

    config = configparser.ConfigParser()
    config.read(f'{input_path}/{config_filename}.ini')

    target_num = int(config['DEFAULT']['NUMBER_OF_TARGETS'])
    scaler = int(config['DEFAULT']['SCALER'])

    if mrs3_mode == EDSR:
        upscaled = _upscale_by_edsr(f'{input_path}/{downscaled_filename}.png', scaler=scaler)
    else:
        upscaled = _upscale_by_resize(f'{input_path}/{downscaled_filename}.png', scaler=scaler, interpolation=mrs3_mode)
    restored = upscaled.copy()

    for i in range(target_num):
        y_from, y_to, x_from, x_to = int(config[f'{i}']['Y_FROM']), int(config[f'{i}']['Y_TO']), int(config[f'{i}']['X_FROM']), int(config[f'{i}']['X_TO'])
        roi = cv2.imread(f'{input_path}/{roi_filename}{i}.png')
        roi_mask = cv2.imread(f'{input_path}/{roi_binary_filename}{i}.png')

        bool_roi_mask_3ch = roi_mask > 0
        bool_roi_mask_1ch = np.all(roi_mask != [0, 0, 0], axis=2)
        bin_roi_mask = bool_roi_mask_1ch.astype(np.uint8) * 255

        # 바이너리 이미지 흰 영역은 원본 부분 선택, 검은 부분은 upscaled 된 부분 선택
        combined_roi = np.where(bool_roi_mask_3ch, roi, upscaled[y_from:y_to, x_from:x_to])
        restored[y_from:y_to, x_from:x_to] = combined_roi

        contours, _ = cv2.findContours(bin_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        restored[y_from:y_to, x_from:x_to] = _blend_images_with_contour_distance(upscaled[y_from:y_to, x_from:x_to], restored[y_from:y_to, x_from:x_to], contours[0], blend=BLEND_SINUSOIDAL)

    cv2.imshow('restored img', restored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 폴더 설정 안 하면 현재 폴더에 바로 이미지 생성
    if output_path == "":
        cv2.imwrite(f'{restored_filename}.png', restored)
        return

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(f'{output_path}/{restored_filename}.png', restored) # [cv2.IMWRITE_PNG_COMPRESSION, 9]
    return


# 용량이 큰 이미지 분할해서 upscaling 할 수 있도록 구현
# 분할된 부분이 겹치도록 - 겹친 후에 블렌딩으로 자연스럽게
PIXELS_LIMIT = 281000
OVERLAP_HALF_LENGTH = 20

def upscale_large_img(img, scaler):
    upsampled_fraction_num = 0
    
    def upscale_img(_img):
        nonlocal upsampled_fraction_num
        h, w, c = _img.shape
        if h * w < PIXELS_LIMIT:
            upsampled_fraction_num += 1
            return sr.upsample(_img)
        
        if w < h:
            upper = np.zeros((scaler*h, scaler*w, c))
            below = np.zeros((scaler*h, scaler*w, c))

            upper[0:scaler*(h//2 + OVERLAP_HALF_LENGTH),:] = upscale_img(_img[0:(h//2 + OVERLAP_HALF_LENGTH),:])
            below[scaler*(h//2 - OVERLAP_HALF_LENGTH):scaler*h,:] = upscale_img(_img[(h//2 - OVERLAP_HALF_LENGTH):h,:])

            # h//2 - OVERLAP_HALF_LENGTH ~ h//2 + OVERLAP_HALF_LENGTH
            alpha = np.ones((scaler*h, scaler*w)) * np.arange(scaler * h).reshape(-1, 1)
            start = scaler * (h//2 - OVERLAP_HALF_LENGTH)
            end = scaler * (h//2 + OVERLAP_HALF_LENGTH)
            alpha = np.clip((alpha - start) / (end - start), 0, 1)

            alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            upper_f = upper.astype(np.float32)
            below_f = below.astype(np.float32)

            blended = upper_f * (1-alpha_3ch) + below_f * alpha_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return blended
        else:
            left = np.zeros((scaler*h, scaler*w, c))
            right = np.zeros((scaler*h, scaler*w, c))
            
            left[:,0:scaler*(w//2 + OVERLAP_HALF_LENGTH)] = upscale_img(_img[:,0:(w//2 + OVERLAP_HALF_LENGTH)])
            right[:,scaler*(w//2 - OVERLAP_HALF_LENGTH):scaler*w] = upscale_img(_img[:,(w//2 - OVERLAP_HALF_LENGTH):w])

            # w//2 - OVERLAP_HALF_LENGTH ~ w//2 + OVERLAP_HALF_LENGTH
            alpha = np.ones((scaler*h, scaler*w)) * np.arange(scaler * w).reshape(1, -1)
            start = scaler * (w//2 - OVERLAP_HALF_LENGTH)
            end = scaler * (w//2 + OVERLAP_HALF_LENGTH)
            alpha = np.clip((alpha - start) / (end - start), 0, 1)
            
            alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            left_f = left.astype(np.float32)
            right_f = right.astype(np.float32)

            blended = left_f * (1-alpha_3ch) + right_f * alpha_3ch
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return blended
        
    if scaler not in [2, 3, 4]:
        print(f"Invalid scaler value: {scaler}. Must be 2, 3 or 4.")
        return None
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-enabled GPU found.")
        return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel(f'models/EDSR_x{scaler}.pb')
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

    # gpu acceleration
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    sr.setModel('edsr', scaler)

    t1 = time.time()
    result = upscale_img(img)
    t2 = time.time()

    if upsampled_fraction_num > 1:
        print(f'upscaled after being divided into {upsampled_fraction_num} fragments.')
    else:
        print(f'upscaled without fraction')
    print(f'{t2-t1} sec taken')

    del sr
    gc.collect()
    return result


# TODO: np 대신 cp 적용 -> 메모리 해제 시간 고려 효율은 생각해봐야함
# TODO: 분할 단위를 2의 거듭제곱으로 할 때, 한계 크기의 절반 보다 약간 크게 잘리면 비효율적 -> 자르는 단위 수정 고려


def compress_img_pkg(img_path, output_path, filename='mrs_output.pkg', scaler=4, roi_mode=ROI_POLYGON, interpolation=INTER_AREA):
    """
    img_path: mrs3 적용할 이미지 경로
    output_path: 결과 저장할 폴더 경로
    filename: 패키징할 파일 이름(e.g. img.pkg)
    scaler: 이미지 downscaling 배율
    roi_mode: 타겟 설정 방식(직사각형, 다각형, 곡선)
    interpolation: downscale 시에 사용할 interpolation
    """

    compress_img_mult_tgs(img_path=img_path, 
                          output_path=output_path, 
                          scaler=scaler,
                          roi_mode=roi_mode, 
                          interpolation=interpolation)
    
    config = configparser.ConfigParser()
    config.read(f'{output_path}/{config_filename}.ini')
    target_num = int(config['DEFAULT']['NUMBER_OF_TARGETS'])
    
    pkg_files = [f'{output_path}/{config_filename}.ini', f'{output_path}/{downscaled_filename}.png']
    for i in range(target_num):
        pkg_files.append(f'{output_path}/{roi_filename}{i}.png')
        pkg_files.append(f'{output_path}/{roi_binary_filename}{i}.png')

    utils.pack_files(output_file=f'{output_path}/{filename}', input_files=pkg_files)

    filesize_pkg = os.path.getsize(f'{output_path}/{filename}')
    print(f'pkg file: {filesize_pkg}')

    for pfile in pkg_files:
        os.unlink(pfile)
    
    return


def restore_img_pkg(input_path, mrs3_mode, output_path=""):
    pass

# TODO: npz 랑 png 압축 비율 비교해봐야함
def compress_img_npz(img_path, output_path, scaler, roi_mode, interpolation=INTER_AREA):
    # TODO: pkg 말고 npz 등으로 저장해도 될듯 -> 압축효율도 높고 -> 다시보니 png 가 이미지에 대해 효율이 높으므로 다수 png 및 pkg 로 처리하는게 나아보인다
    # 우선 png 및 pkg 로 처리하고 나중에 비교해보는 식으로
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error loading image: {img_path}")
        return
    
    targets = _select_multiple_polygon_roi(img_path)

    # TODO: 다양한 interpolation 비교 및 복원 비교
    downscaled = _downscale_img(img_path, scaler, interpolation=interpolation)

    # 각 타겟 위치정보 저장
    config = np.zeros(shape=(len(targets), 4), dtype=np.int8)
    for i in range(len(targets)):
        config[i][0] = targets[i][2][0] # Y_FROM
        config[i][1] = targets[i][2][1] # Y_TO
        config[i][2] = targets[i][2][2] # X_FROM
        config[i][3] = targets[i][2][3] # X_TO

    # 결과저장 폴더 없을 때 새로 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pass



def compress_img_pkg_imgpresso(img_path, output_path, filename='mrs_output.pkg', scaler=4, roi_mode=ROI_POLYGON, interpolation=INTER_AREA):
    """
    압축은 더욱 효과적이지만, 인터넷 접속, 서버 내 처리 및 다운로드 등으로 인한 시간이 추가적으로 소요됨
    img_path: mrs3 적용할 이미지 경로
    output_path: 결과 저장할 폴더 경로
    filename: 패키징할 파일 이름(e.g. img.pkg)
    scaler: 이미지 downscaling 배율
    roi_mode: 타겟 설정 방식(직사각형, 다각형, 곡선)
    interpolation: downscale 시에 사용할 interpolation
    """

    compress_img_mult_tgs(img_path=img_path, 
                          output_path=output_path, 
                          scaler=scaler,
                          roi_mode=roi_mode, 
                          interpolation=interpolation)
    
    config = configparser.ConfigParser()
    config.read(f'{output_path}/{config_filename}.ini')
    target_num = int(config['DEFAULT']['NUMBER_OF_TARGETS'])

    # TODO: utils.compress_imgpresso 사용해서 한 번 더 압축 -> 다만 roi 및 bin 이 다르게 보정되면 복원 시에 검은 점이나 흐릿한 부분이 생길 수 있음. 
    utils.compress_imgpresso_replace(f'{output_path}/{downscaled_filename}.png', output_path)
    for i in range(target_num):
        utils.compress_imgpresso_replace(f'{output_path}/{roi_filename}{i}.png', output_path)
        # utils.compress_imgpresso_replace(f'{output_path}/{roi_binary_filename}{i}.png', output_path) # 바이너리는 적용이 안 되고, 된다하더라도 어차피 바이너리는 용량 작아서 별 의미 없음
    
    pkg_files = [f'{output_path}/{config_filename}.ini', f'{output_path}/{downscaled_filename}.png']
    for i in range(target_num):
        pkg_files.append(f'{output_path}/{roi_filename}{i}.png')
        pkg_files.append(f'{output_path}/{roi_binary_filename}{i}.png')

    utils.pack_files(output_file=f'{output_path}/{filename}', input_files=pkg_files)

    filesize_pkg = os.path.getsize(f'{output_path}/{filename}')
    print(f'pkg file: {filesize_pkg}')

    for pfile in pkg_files:
        os.unlink(pfile)
    
    return

