import cv2
import argparse
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline

def face_fusion(source_path, target_path, output_path):
    image_face_fusion = pipeline('face_fusion_torch',
                                model='damo/cv_unet_face_fusion_torch', 
                                model_revision='v1.0.3')
                                
    result = image_face_fusion(dict(template=source_path, user=target_path))                            
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Fusion')
    parser.add_argument('-s', '--source', required=True, help='source image path')
    parser.add_argument('-t', '--target', required=True, help='target image path')
    parser.add_argument('-o', '--output', required=True, help='output image path')
    
    args = parser.parse_args()
    face_fusion(args.source, args.target, args.output)
