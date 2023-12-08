import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class image_processing:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.net = cv2.dnn.readNet('opencv_face_detector.pbtxt','opencv_face_detector_uint8.pb')

    def img_preprocessing(self):
        # 이미지 읽기
        img = cv2.imread(self.image_path)
        # 그레이스케일 변환
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 가우시안 블러 적용
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # 케니 엣지 검출
        edges = cv2.Canny(blurred_image, 50, 150)
        # 컨투어 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 컨투어 중에서 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        # 주민등록증 영역 추출
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = img[y:y+h, x:x+w]
        result_path = "uploads/preprocessing.png"
        cv2.imwrite(result_path,cropped_image)
        return cropped_image

    def find_text_indices(self, df):
        input_text = ''.join([df['word'][i] for i in range(len(df))])
        # 정규표현식 패턴
        pattern = re.compile(r'.{5}(.*?)(?=\()')

        match = pattern.search(input_text)
        if match:
            end_index = match.end()
        selected_rows = df.iloc[end_index-2]
        result_array = selected_rows[['1','2','3','4']].values
        return result_array

    def find_chi(self, df):
        input_text = ''.join([df['word'][i] for i in range(len(df))])
        start_index = input_text.find('(')
        selected_rows = df.loc[start_index+2]
        result_array = selected_rows[['1', '2', '3', '4']].values
        return result_array

    def find_address_after_gu(self, df):
        # 주소 추출을 위한 정규표현식 패턴
        input_text = ''.join([df['word'][i] for i in range(len(df))])
        address_pattern = re.compile(r'([^\d]+시[^\d]+구)')

        # 정규표현식과 매치되는 부분 찾기
        matches = address_pattern.search(input_text)

        # 매치된 문자열이 있을 경우 해당 문자열의 시작과 끝 인덱스 반환
        if matches:
            start_index = matches.end(0)  # "구" 이후의 인덱스
            end_index = input_text.find(")", start_index) + 1  # ")"의 다음 인덱스
            selected_rows = df.iloc[start_index:end_index]
            result_array = selected_rows[['1','2','3','4']].values
            return result_array
        else:
            return None

    def detect_rrn(self, df):
        # DataFrame의 'word' 열 문자열을 합침
        string = ''.join([df['word'][i] for i in range(len(df))])
        # 정규표현식 패턴
        pattern = re.compile(r'-(\d{1})(\d{6})')

        # 문자열에서 매치된 부분 찾기
        matches = pattern.finditer(string)

        # 매치된 결과 출력
        result_array = []
        for match in matches:
            start_index = match.start() + 2
            end_index = match.end()

            # DataFrame에서 해당 범위의 행 선택
            selected_rows = df.iloc[start_index:end_index]

            # 선택된 행의 인덱스를 배열로 추가
            result_array = selected_rows[['1','2','3','4']].values
        return result_array

    def process_image(self, resized_img):
        # 이미지 로드
        locations = []
        # 이미지가 정상적으로 로드되었는지 확인
        if resized_img is None:
            print(f"Error: Unable to load image from {self.image_path}")
            return
        # 얼굴 감지 모델 로드
        h, w = resized_img.shape[:2]
        
        # 이미지 전처리
        blob = cv2.dnn.blobFromImage(cv2.resize(resized_img, (300, 300)), 1.0, (300, 300), (123.68, 116.78, 103.94))
        
        # 모델 입력 설정
        self.net.setInput(blob)
        detections = self.net.forward()

        # 얼굴 감지 결과
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)

                # 얼굴에 사각형 표시
                cv2.rectangle(resized_img, (startX, startY), (endX, endY), (255, 255, 255), 2)
                resized_img[startY:endY, startX:endX] = [255, 255, 255]
        locations = []
        # 이미지가 정상적으로 로드되었는지 확인
        if resized_img is None:
            print(f"Error: Unable to load image from {self.image_path}")
        else:
            # 그레이스케일 변환
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            # 추출된 텍스트의 위치 찾기
            digits_loc = pytesseract.image_to_boxes(gray, lang='kor+chi_sim', config='--oem 3 --psm 6')
            for digit_loc in digits_loc.splitlines():
                digit_loc = digit_loc.split()
                locations.append(digit_loc)
        #=====================================================
        df = pd.DataFrame(locations)
        df.columns = ['word','1','2','3','4','5']
        df = df.drop(['5'], axis=1)

        hello = []
        hello.append(self.find_text_indices(df))
        hello.append(self.find_chi(df))
        hello.append(self.find_address_after_gu(df))
        hello.append(self.detect_rrn(df))
        hello = np.vstack(hello)
        # 이미지에 네모 박스로 마스킹하기
        for box in hello:
                x,y,w,h = box[0],box[1],box[2],box[3]
                x, y, w, h = int(x), int(y), int(w), int(h)

                y = resized_img.shape[0] - y
                h = resized_img.shape[0] - h

                cv2.rectangle(resized_img, (x, y), (w, h), 5, -1)
        return resized_img

    def process_document(self):
        # 이미지 전처리
        resized_img = self.img_preprocessing()
        # 마스킹 처리
        result_img = self.process_image(resized_img)
        result_path = "uploads/result.png"
        cv2.imwrite(result_path, result_img)
        # 결과 이미지 파일의 경로를 반환
        return result_path


# 예제 사용:
image_path = 'min.png'  # 실제 이미지 경로로 대체하세요

doc_processor = image_processing(image_path)
doc_processor.process_document()
