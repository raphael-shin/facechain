import cv2
import argparse
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline

def face_fusion(user_path, template_path, output_path):
    image_face_fusion = pipeline('face_fusion_torch',
                                model='damo/cv_unet_face_fusion_torch', 
                                model_revision='v1.0.3')
                                
    result = image_face_fusion(dict(template=template_path, user=user_path))                            
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Fusion')
    parser.add_argument('-u', '--user_path', required=True, help='user image path')
    parser.add_argument('-t', '--template_path', required=True, help='template_path image path')
    parser.add_argument('-o', '--output_path', required=True, help='output image path')
    
    args = parser.parse_args()
    face_fusion(args.user_path, args.template_path, args.output_path)
