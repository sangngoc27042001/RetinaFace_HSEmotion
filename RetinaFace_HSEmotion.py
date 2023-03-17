from config import config
from RetinaFaceInterference import detect_face_retinaface
from hsemotion.facial_emotions import HSEmotionRecognizer



fer=HSEmotionRecognizer(model_name=config['HSEmotionRecognizer']['weight'],device=config['device'])

def get_bboxes_emotion(img_rgb):
    # s = time.time()
    bounding_boxes=detect_face_retinaface(img_rgb)
    if bounding_boxes is None:
        return None
    # print(time.time()-s)
    # s = time.time()
    face_img_list = [img_rgb[y1:y2,x1:x2,:] for x1,y1,x2,y2 in [box[0:4] for box in bounding_boxes]]
    emotions,scores=fer.predict_multi_emotions(face_img_list,logits=False)
    # print(time.time()-s)
    # s = time.time()
    return list(zip(bounding_boxes, emotions))


