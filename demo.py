from RetinaFace_HSEmotion import get_bboxes_emotion
import argparse
import os
import cv2



def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

def handle_one_image(img_path, args):
    img = cv2.imread(img_path)
    frame_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        bbs_em = get_bboxes_emotion(frame_in)
    except:
        bbs_em = None

    no_faces = 0 if bbs_em is None else len(bbs_em)
    print(f'{img_path} [DONE] [{no_faces} {"face" if no_faces<2 else "faces"} detected]')

    if no_faces >0:
        for bb, em in bbs_em:
            l, t, r, b = bb
            cv2.rectangle(img, (l, t), (r, b), (36,255,12), 5)   
            cv2.putText(img, f'{em}', (l,t),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
    os.makedirs(args.output, exist_ok=True)
    file_name= os.path.basename(img_path)
    cv2.imwrite(f'./{args.output}/{file_name}',img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="The path of input folder containing images")
    parser.add_argument('--output', type=str, help="The path of output folder")
    args = parser.parse_args()
    [handle_one_image(os.path.join(args.input, img), args) for img in os.listdir(args.input)]
    print(f'All images have been saved in folder: {args.output}')

    
    
