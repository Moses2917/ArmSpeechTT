import cv2
import speech_recognition as sr
import threading
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from transformers import pipeline

recognizer = sr.Recognizer()
video = cv2.VideoCapture(0)
text = ""
def record():
    while True:
        global text
        check, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#Have to do a lot of converting here, because cv2 does not natively support armenian chars ie: unicode
        pil_image = Image.fromarray(image)
        
        # Draw non-ascii text onto image
        font = ImageFont.truetype("C:\Windows\Fonts\\arial.ttf", 50) #add a cycling text str so the new is on the bottom and old on top, use stack/queue
        draw = ImageDraw.Draw(pil_image)
        draw.text((30, 30), text, font=font,stroke_width=3,stroke_fill=(0,0,0))
        image = np.asarray(pil_image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Armenian subtitles", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    
 
def voice():
    global text
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="Chillarmo/whisper-small-hy-AM",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        device=0,
        generate_kwargs={"language":"armenian"}
    )
    while True:
        try:
            with sr.Microphone() as source:
                print("recording")
                recorded_audio = recognizer.listen(source, timeout=3,phrase_time_limit=7)
                # wave.open(/)
                try:
                    text = ""
                    print("Analyzing the audio...")
                    result = pipe(recorded_audio.get_flac_data())
                    print(result["text"])
                    text = result["text"]
                except:
                    print("nothing found")
                    text = ""
        except:
            print("no voice")
t1 = threading.Thread(target=record, args=[])
t2 = threading.Thread(target=voice, args=[])
t1.start()
t2.start()
