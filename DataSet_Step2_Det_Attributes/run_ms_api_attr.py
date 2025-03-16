import os
import time
import json
import pickle
import argparse
#from azure.cognitiveservices.vision.face import FaceClient
#from msrest.authentication import CognitiveServicesCredentials
import boto3
from PIL import Image

# Put your KEY and ENDPOINT here
#KEY = "Your KEY"
#ENDPOINT = "Your ENDPOINT"
#face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

face_client = boto3.client('rekognition')

def prase_face(face):
    res = {}
    #attr = face.face_attributes
    age_range = face.get('AgreRange', {})
    res['Age'] = (age_range.get('Low', 0) + age_range.get('High', 0)) / 2.0 if age_range else 0.0
    gender_str = face.get('Gender', {}).get('Value', '').lower()
    res['Gender'] = 1.0 if gender_str == 'male' else 0.0 
    res['Expression'] = float(face.get('Emotions', [{}])[0].get('Confidence', 0))
    res['Glasses'] = 1.0 if face.get('Eyeglasses', {}).get('Value', False) else 0.0
    res['Yaw'] = float(face.get('Pose', {}).get('Yaw', 0))
    res['Pitch'] = float(face.get('Pose', {}).get('Pitch', 0))
    hair = face.get('Hair', {})
    res['Baldness'] = float(hair.get('Bald', 0.0))
    res['Beard'] = float(face.get('Beard', {}).get('Confidence', 0))
    #res['Age'] = attr.age
    #res['Gender'] = 1 if str(attr.gender).split('.')[-1] == 'male' else 0
    #res['Expression'] = attr.smile
    #res['Glasses'] = 0 if str(attr.glasses).split('.')[-1] == 'no_glasses' else 1
   # res['Yaw'] = attr.head_pose.yaw
   # res['Pitch'] = attr.head_pose.pitch
   # res['Baldness'] = attr.hair.bald
   # res['Beard'] = attr.facial_hair.beard if attr.facial_hair else 0.0
    return res


def detect_face(image):

    # The required face attributes
    #required_attr = [
    #    'age',
    #    'gender',
    #    'headPose',
    #    'smile',
    #    'facialHair',
    #    'glasses',
    #    'emotion',
    #    'hair',
    #    'makeup',
    #    'occlusion',
    #    'accessories',
    #    'blur',
    #    'exposure',
    #    'noise',
    #    'qualityForRecognition',
    #]

    # We use detection model 3 to get better performance
    # recognition model 4 to support quality for recognition attribute.
    #faces = face_client.face.detect_with_stream(image,
   #                                             detection_model='detection_01',
   #                                             recognition_model='recognition_03',
   #                                             return_face_attributes=required_attr)
#
#    return faces
    with open(image, 'r+b') as image_file:
        image_bytes = image_file.read()

    response = face_client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
    )

    return response.get('FaceDetails', [])

if __name__ == '__main__':
    '''Usage
    cd ./DataSet_Step2_Det_Attributes
    python run_ms_api_attr.py --proj_data_dir ../examples/dataset_examples
    '''

    parser = argparse.ArgumentParser(description="ms_api_attributes")
    parser.add_argument(
        "--proj_data_dir",
        type=str,
        required=True,
        help="The directory of the project data, which should inculde 'inversions' sub-directory.",
    )
    args = parser.parse_args()

    # ----------------------- Define Inference Parameters -----------------------
    input_images_dir = os.path.join(args.proj_data_dir, 'inversions')
    output_attr_dir = os.path.join(args.proj_data_dir, 'attributes')
    output_ms_api_attr_dir = os.path.join(args.proj_data_dir, 'attributes_ms_api')

    os.makedirs(output_attr_dir, exist_ok=True)
    os.makedirs(output_ms_api_attr_dir, exist_ok=True)

    # ----------------------- MS API Detection -----------------------
    fnames = sorted(os.listdir(input_images_dir))
    for fn in fnames:
        basename = fn[:fn.rfind('.')]

        tic = time.time()

        # retry until the response is done or num_retry larger than 5.
        max_retry = 5
        num_retry = 0
        is_drop = False
        while num_retry <= max_retry:
            try:
                image = os.path.join(input_images_dir, fn)
                ms_api_attr = detect_face(image)
                if len(ms_api_attr) != 1:
                    is_drop = True  # Only need images only with one face
                else:
                    ms_api_attr = ms_api_attr[0]
                break
            except Exception as e:
                num_retry += 1
                print(f'>> Error: {str(e)}.')
                print(f'>> MS API error, sleep 2 minutes and retry {num_retry}-st.')
                time.sleep(120)

        # If there is no face or multiply faces, drop this image.
        if is_drop:
            os.unlink(os.path.join(args.proj_data_dir, 'latents', f"{basename}.pt"))
            os.unlink(os.path.join(args.proj_data_dir, 'inversions', f"{basename}.png"))
            os.unlink(os.path.join(args.proj_data_dir, 'lights', f"{basename}.npy"))
            print(f'>> Drop {fn}, there is no face or multiply faces in the inversed image!')
            continue

        if num_retry > max_retry:
            raise Exception(f'Retry more than {max_retry} times.')

        parsed_attr = prase_face(ms_api_attr)

        with open(os.path.join(output_attr_dir, f'{basename}.json'), 'w') as f:
            json.dump(parsed_attr, f, indent=4, sort_keys=True)
        pickle.dump(ms_api_attr, open(os.path.join(output_ms_api_attr_dir, f'{basename}.pkl'), 'wb'))

        time.sleep(3)

        toc = time.time()
        print('Detect attribute {} took {:.4f} seconds.'.format(fn, toc - tic))

    print('MS API detect attributes done!')
