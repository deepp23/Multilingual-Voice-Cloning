from flask import Flask, render_template, request, send_file
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import librosa
from pydub import AudioSegment
import io
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_name = request.form['name']
    uploaded_file = request.files['file']
    language = request.form['language']
    
    audio_array, sampling_rate = librosa.load(uploaded_file, sr=None)
    audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

    # Process the audio
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    generate_ids = model.generate(
        inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=processor.tokenizer.lang_code_to_id[language],
    )
    translation = processor.batch_decode(generate_ids, skip_special_tokens=True)
    result_string = " ".join(translation)
    print(result_string)

    data = uploaded_file.read()
    print('input data:', data)
    uploaded_file.seek(0) 
    path=".\input.mp3"
    uploaded_file.save(path)

    client = ElevenLabs(
        api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"#Your API Key
    )
    voice = client.clone(
    name=uploaded_name,
    description="A indian male voice",
    files=[path],
    )
    audio = client.generate(text=result_string, voice=voice)

    audio_bytes = b''.join(audio)
    audio_bytes_io = io.BytesIO(audio_bytes)
    audio_segment = AudioSegment.from_file(audio_bytes_io)
    output_voice_path = os.path.join('.', 'output.mp3')
    audio_segment.export(output_voice_path, format="mp3")
    audio_bytes_io.seek(0)
    return send_file(audio_bytes_io, as_attachment=True, mimetype="audio/mpeg", download_name="output.mp3")

if __name__ == '__main__':
    app.run(debug=True)



