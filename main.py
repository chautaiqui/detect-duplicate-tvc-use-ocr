!pip install easyocr
!pip install easyocr python-Levenshtein scikit-learn

import cv2
import easyocr
import numpy as np
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize EasyOCR Reader (with English, Vietnamese language)
reader = easyocr.Reader(['en', 'vi'])

# Define function to apply EasyOCR on the frame
def extract_text_from_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray_frame)
    extracted_text = " ".join([res[1] for res in result])
    return extracted_text.strip()

# Function to combine text from all frames of a TVC
def extract_text_from_tvc(video_path, fps_skip=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    combined_text = []
    
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps * fps_skip)
        ret, frame = cap.read()
        if not ret:
            break
        text = extract_text_from_frame(frame)
        if text:
            combined_text.append(text)
        frame_count += 1
    
    cap.release()
    return " ".join(combined_text)  # Combine all text into a single string

# Function to calculate Levenshtein similarity between two texts
def levenshtein_similarity(text1, text2, threshold=0.80):
    if not text1 or not text2:
        return False  # If either string is empty, they can't be considered the same
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    similarity = 1 - (distance / max_len)  # Normalize the distance to be between 0 and 1
    return similarity >= threshold

# Function to calculate cosine similarity between two texts
def cosine_similarity_texts(text1, text2, threshold=0.80):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)[0][1]  # Get the similarity between the two texts
    print('cos_sim', cos_sim, threshold, cos_sim >= threshold)
    return cos_sim >= threshold

# Main function to check if two TVCs are duplicates
def are_tvc_duplicates(video_path1, video_path2, method="cosine", threshold=0.8, fps_skip=1):
    # Extract combined text from each TVC
    text1 = extract_text_from_tvc(video_path1, fps_skip)
    text2 = extract_text_from_tvc(video_path2, fps_skip)
    print(f"Text1: {text1}")
    print(f"Text2: {text2}")
    # Compare texts using the specified method
    if method == "levenshtein":
        return levenshtein_similarity(text1, text2, threshold)
    elif method == "cosine":
        return cosine_similarity_texts(text1, text2, threshold)
    else:
        raise ValueError("Unknown comparison method")

# Example usage:
video_path1 = 'https://r2---sn-i3b7kns6.c.2mdn.net/videoplayback/id/3d93e805540bcb47/itag/347/source/web_video_ads/xpc/EgVovf3BOg%3D%3D/ctier/L/acao/yes/ip/0.0.0.0/ipbits/0/expire/3865138565/sparams/acao,ctier,expire,id,ip,ipbits,itag,mh,mip,mm,mn,ms,mv,mvi,pl,source,xpc/signature/0C73F3DCF3940B2AA00AFE411BCBEAC90B48866A.6EA8D3D03E770FCC8BD657FD97469BA9CB0BD429/key/cms1/cms_redirect/yes/mh/jp/mip/58.187.123.23/mm/42/mn/sn-i3b7kns6/ms/onc/mt/1726037000/mv/u/mvi/2/pl/24/file/file.mp4'
video_path2 = 'https://r2---sn-i3b7kns6.c.2mdn.net/videoplayback/id/3d93e805540bcb47/itag/692/source/web_video_ads/xpc/EgVovf3BOg%3D%3D/ctier/L/acao/yes/ip/0.0.0.0/ipbits/0/expire/3865138565/sparams/acao,ctier,expire,id,ip,ipbits,itag,mh,mip,mm,mn,ms,mv,mvi,pl,source,xpc/signature/5E85BC6BE30E604925EA634FF0BDEF3BACA8A7EF.2EB7BDEB53AD11A3441E2C6B7E5C596B2E6B5C88/key/cms1/cms_redirect/yes/mh/jp/mip/58.187.123.23/mm/42/mn/sn-i3b7kns6/ms/onc/mt/1726037000/mv/u/mvi/2/pl/24/file/file.mp4'

# Check if the two TVCs are duplicates using cosine similarity
is_duplicate = are_tvc_duplicates(video_path1, video_path2, method="cosine", threshold=0.80)
print(f"Are the two TVCs duplicates? {is_duplicate}")
