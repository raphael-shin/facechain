from modelscope.pipelines import pipeline

def download_model():
    # 모델 초기화만 수행하여 다운로드 트리거
    pipeline('face_fusion_torch',
             model='damo/cv_unet_face_fusion_torch',
             model_revision='v1.0.3')

if __name__ == '__main__':
    download_model()
