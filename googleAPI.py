def run_label(file_name): // 라벨 추출
    import io
    import os
 
    from google.cloud import vision
    from google.cloud.vision import types
 
    client = vision.ImageAnnotatorClient()
    
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
 
    image = types.Image(content=content)
 
    response = client.label_detection(image=image)
    labels = response.label_annotations

    
    for label in labels:
        print(label)  
    
if __name__ == '__main__':
    run_label("news.jpg")
    
    

def run_ocr(file_name): // text 추출
    import io
    import os
 
    # 구글 라이브러리 import
    from google.cloud import vision
    from google.cloud.vision import types
 
    # 사용할 클라이언트 설정
    client = vision.ImageAnnotatorClient()
    
    # 이미지 읽기
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
 
    image = types.Image(content=content)
    
    response_ocr= client.text_detection(image=image)
    
    texts=response_ocr.text_annotations
    
    textocr = texts[0].description
    
    print(textocr)
    
    
if __name__ == '__main__':
    run_ocr("ocr1.png")


from google.cloud import translate_v2 as translate // 해석

# Instantiates a client
translate_client = translate.Client()

# The text to translate
text = 'Hello, Korea!'

translation = translate_client.translate(
    text,
    target_language='ko')

translation2 = translate_client.translate(
    text,
    target_language='ja')
    
print(translation)

print(translation2)

