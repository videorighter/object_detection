# object_detection

detectron2를 이용한 object detection입니다.

인스타그램 "롬앤립스틱" 해시태그의 사진을 크롤링하여 분석에 사용했습니다.

task는 사진에서 얼굴(face), 입술(lip), 제품(product)을 detection 하여 화장품 사용자의 발색샷, 실사용샷을 구분하는 task입니다.

https://github.com/cgvict/roLabelImg

labeling toolkit인 rolabelImg를 통해 x, y, h, w를 측정하여 xml 파일로 만든 뒤,

json 파일로 만들어 학습에 사용했습니다.

val_result에 detection 결과가 있으나 얼굴 사진은 문제 발생을 우려하여 포함하지 않았습니다.
