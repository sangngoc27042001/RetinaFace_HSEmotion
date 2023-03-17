config = {
    'device':'cpu',
    'RetinaFaceInterference':{
        'conf_thres':0.5 #the threshold of face detection
    },
    'HSEmotionRecognizer':{
        'weight': 'enet_b0_8_best_afew', #['enet_b0_8_best_afew', 'enet_b0_8_best_vgaf', 'enet_b0_8_va_mtl', 'enet_b2_8']
    }
}