
# Multilingual-Voice-Cloning

This project lets you talk in one language, and it changes what you say into another language while still sounding like you. So, if you speak English, it can turn your words into French or any other language you choose, but it will still sound like your voice. 

This project use various pre-trained Large Language Models such as facebook s2t model and Eleven Labs API.

This project is developed using Flask, JavaScript, HTML, and CSS. The Python code responsible for the backend functionality is explained below:




## Deployment

Note: Please note that when using the app.route function in Flask, it's essential to combine all relevant code cells to ensure they run seamlessly together. This ensures that the routes, handlers, and associated functionality are properly integrated and executed within the Flask application.

Before we begin, let's ensure that all the necessary packages are installed. You can easily do this by running the following command in your terminal:

```bash
  pip install -r requirements.txt
```
This command will automatically install all the required packages listed in the requirements.txt file, ensuring that your environment is properly set up for the project. Once the installation is complete, you're ready to proceed with running the code in your preferred code editor.

And then import the following libraries. I'll explain what each of these libraries do in the coming steps.
```bash
from flask import Flask, render_template, request, send_file
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import librosa
from pydub import AudioSegment
import io
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
```

We then load the facebook s2t model and tokenizer by using HuggingFace library for converting the audio into text.
```bash
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
```
This code creates an instance of Flask
```bash
app = Flask(__name__)
```

```bash
@app.route('/')
def index():
    return render_template('base.html')
```
The part of code, enclosed within the @app.route decorator for the /upload endpoint and the upload_file function definition, handles the POST request for file uploads. It extracts the uploaded file's name, file object, and selected language from the request form data, facilitating further processing of the uploaded content.
```bash
@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_name = request.form['name']
    uploaded_file = request.files['file']
    language = request.form['language']
```
Don't run these like cells in notebook, for sake for explaining the code I have divided the code but run the code as a whole.

We use the librosa library to extract and manipulate the features like audio_array, sampling_rate from the uploaded_file and send it to the processor(tokenizer) into respective way which is suitable for passing it through the s2t model. We also give forced_bos_token_id as parameter which specifies the language the text is to be generated. Here we send language as we take input from the HTML form.
```bash
    audio_array, sampling_rate = librosa.load(uploaded_file, sr=None)
    audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    generate_ids = model.generate(
        inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=processor.tokenizer.lang_code_to_id[language],
    )
    translation = processor.batch_decode(generate_ids, skip_special_tokens=True)
    result_string = " ".join(translation)
    print(result_string)
```
This is one of the important part of code and almost everyone gets this wrong. Whenever we take the uploaded_file from HTML form, we cannot directly use in most of the cases as it is FileStorage Object. It works well with librosa library but ElevenLabs doesn't support. So we save the file in the disk using .save() and we set the .seek(0) so the pointer points back to beginning of the data or else if it is at the end we won't be able to store any data in the disk.
```bash
    data = uploaded_file.read()
    print('input data:', data)
    uploaded_file.seek(0) 
    path=".\\output\\input.mp3"
    uploaded_file.save(path)
```
Here, we initialize an instance of the ElevenLabs client, providing the required API key for authentication. Subsequently, we create a new voice clone using the clone method, specifying attributes such as the name of the voice and a description. The audio file path is included in the list of files to be used for voice cloning. Finally, we generate the audio output using the specified text and voice parameters.

```bash
    client = ElevenLabs(
api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"#Your API Key
    )
    voice = client.clone(
    name=uploaded_name,
    description="A indian male voice",
    files=[path],
    )
    audio = client.generate(text=result_string, voice=voice)
```
In this final section, we convert the audio data into a BytesIO object named audio_bytes_io, which is then used to initialize an AudioSegment. This segment is created from the audio data contained in the BytesIO object. Subsequently, the audio segment is exported to an MP3 file named output.mp3 and saved in the output directory. Finally, we reset the position of the BytesIO object to the beginning and return it as a downloadable attachment using the send_file function, with the filename set to output.mp3.

```bash
audio_bytes_io = io.BytesIO(audio)
    audio_segment = AudioSegment.from_file(audio_bytes_io)
    output_voice_path = os.path.join('.','output.mp3')
    audio_segment.export(output_voice_path, format="mp3")
    audio_bytes_io.seek(0)  
    return send_file(audio_bytes_io, as_attachment=True, attachment_filename='output.mp3')
```
This code starts exeuting the whole code from the beginning.

```bash
if __name__ == '__main__':
    app.run(debug=True)
```
## Appendix

Links for more information

https://elevenlabs.io/docs/api-reference/python-text-to-speech-guide

https://flask.palletsprojects.com/en/3.0.x/

https://huggingface.co/facebook/s2t-medium-mustc-multilingual-st

## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/palthyadeepmalik)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/Deepmalik177)
