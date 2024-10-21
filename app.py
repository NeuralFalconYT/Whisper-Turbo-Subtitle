#@title Utils
language_dict = {
    "Akan": {"lang_code": "aka", "meta_code": "aka_Latn"},
    "Albanian": {"lang_code": "sq", "meta_code": "als_Latn"},
    "Amharic": {"lang_code": "am", "meta_code": "amh_Ethi"},
    "Arabic": {"lang_code": "ar", "meta_code": "arb_Arab"},
    "Armenian": {"lang_code": "hy", "meta_code": "hye_Armn"},
    "Assamese": {"lang_code": "as", "meta_code": "asm_Beng"},
    "Azerbaijani": {"lang_code": "az", "meta_code": "azj_Latn"},
    "Basque": {"lang_code": "eu", "meta_code": "eus_Latn"},
    "Bashkir": {"lang_code": "ba", "meta_code": "bak_Cyrl"},
    "Bengali": {"lang_code": "bn", "meta_code": "ben_Beng"},
    "Bosnian": {"lang_code": "bs", "meta_code": "bos_Latn"},
    "Bulgarian": {"lang_code": "bg", "meta_code": "bul_Cyrl"},
    "Burmese": {"lang_code": "my", "meta_code": "mya_Mymr"},
    "Catalan": {"lang_code": "ca", "meta_code": "cat_Latn"},
    "Chinese": {"lang_code": "zh", "meta_code": "zh_Hans"},
    "Croatian": {"lang_code": "hr", "meta_code": "hrv_Latn"},
    "Czech": {"lang_code": "cs", "meta_code": "ces_Latn"},
    "Danish": {"lang_code": "da", "meta_code": "dan_Latn"},
    "Dutch": {"lang_code": "nl", "meta_code": "nld_Latn"},
    "English": {"lang_code": "en", "meta_code": "eng_Latn"},
    "Estonian": {"lang_code": "et", "meta_code": "est_Latn"},
    "Faroese": {"lang_code": "fo", "meta_code": "fao_Latn"},
    "Finnish": {"lang_code": "fi", "meta_code": "fin_Latn"},
    "French": {"lang_code": "fr", "meta_code": "fra_Latn"},
    "Galician": {"lang_code": "gl", "meta_code": "glg_Latn"},
    "Georgian": {"lang_code": "ka", "meta_code": "kat_Geor"},
    "German": {"lang_code": "de", "meta_code": "deu_Latn"},
    "Greek": {"lang_code": "el", "meta_code": "ell_Grek"},
    "Gujarati": {"lang_code": "gu", "meta_code": "guj_Gujr"},
    "Haitian Creole": {"lang_code": "ht", "meta_code": "hat_Latn"},
    "Hausa": {"lang_code": "ha", "meta_code": "hau_Latn"},
    "Hebrew": {"lang_code": "he", "meta_code": "heb_Hebr"},
    "Hindi": {"lang_code": "hi", "meta_code": "hin_Deva"},
    "Hungarian": {"lang_code": "hu", "meta_code": "hun_Latn"},
    "Icelandic": {"lang_code": "is", "meta_code": "isl_Latn"},
    "Indonesian": {"lang_code": "id", "meta_code": "ind_Latn"},
    "Italian": {"lang_code": "it", "meta_code": "ita_Latn"},
    "Japanese": {"lang_code": "ja", "meta_code": "jpn_Jpan"},
    "Kannada": {"lang_code": "kn", "meta_code": "kan_Knda"},
    "Kazakh": {"lang_code": "kk", "meta_code": "kaz_Cyrl"},
    "Korean": {"lang_code": "ko", "meta_code": "kor_Hang"},
    "Kurdish": {"lang_code": "ckb", "meta_code": "ckb_Arab"},
    "Kyrgyz": {"lang_code": "ky", "meta_code": "kir_Cyrl"},
    "Lao": {"lang_code": "lo", "meta_code": "lao_Laoo"},
    "Lithuanian": {"lang_code": "lt", "meta_code": "lit_Latn"},
    "Luxembourgish": {"lang_code": "lb", "meta_code": "ltz_Latn"},
    "Macedonian": {"lang_code": "mk", "meta_code": "mkd_Cyrl"},
    "Malay": {"lang_code": "ms", "meta_code": "ms_Latn"},
    "Malayalam": {"lang_code": "ml", "meta_code": "mal_Mlym"},
    "Maltese": {"lang_code": "mt", "meta_code": "mlt_Latn"},
    "Maori": {"lang_code": "mi", "meta_code": "mri_Latn"},
    "Marathi": {"lang_code": "mr", "meta_code": "mar_Deva"},
    "Mongolian": {"lang_code": "mn", "meta_code": "khk_Cyrl"},
    "Nepali": {"lang_code": "ne", "meta_code": "npi_Deva"},
    "Norwegian": {"lang_code": "no", "meta_code": "nob_Latn"},
    "Norwegian Nynorsk": {"lang_code": "nn", "meta_code": "nno_Latn"},
    "Pashto": {"lang_code": "ps", "meta_code": "pbt_Arab"},
    "Persian": {"lang_code": "fa", "meta_code": "pes_Arab"},
    "Polish": {"lang_code": "pl", "meta_code": "pol_Latn"},
    "Portuguese": {"lang_code": "pt", "meta_code": "por_Latn"},
    "Punjabi": {"lang_code": "pa", "meta_code": "pan_Guru"},
    "Romanian": {"lang_code": "ro", "meta_code": "ron_Latn"},
    "Russian": {"lang_code": "ru", "meta_code": "rus_Cyrl"},
    "Serbian": {"lang_code": "sr", "meta_code": "srp_Cyrl"},
    "Sinhala": {"lang_code": "si", "meta_code": "sin_Sinh"},
    "Slovak": {"lang_code": "sk", "meta_code": "slk_Latn"},
    "Slovenian": {"lang_code": "sl", "meta_code": "slv_Latn"},
    "Somali": {"lang_code": "so", "meta_code": "som_Latn"},
    "Spanish": {"lang_code": "es", "meta_code": "spa_Latn"},
    "Sundanese": {"lang_code": "su", "meta_code": "sun_Latn"},
    "Swahili": {"lang_code": "sw", "meta_code": "swa_Latn"},
    "Swedish": {"lang_code": "sv", "meta_code": "swe_Latn"},
    "Tamil": {"lang_code": "ta", "meta_code": "tam_Taml"},
    "Telugu": {"lang_code": "te", "meta_code": "tel_Telu"},
    "Thai": {"lang_code": "th", "meta_code": "tha_Latn"},
    "Turkish": {"lang_code": "tr", "meta_code": "tur_Latn"},
    "Ukrainian": {"lang_code": "uk", "meta_code": "ukr_Cyrl"},
    "Urdu": {"lang_code": "ur", "meta_code": "urd_Arab"},
    "Uzbek": {"lang_code": "uz", "meta_code": "uzb_Latn"},
    "Vietnamese": {"lang_code": "vi", "meta_code": "vie_Latn"},
    "Welsh": {"lang_code": "cy", "meta_code": "cym_Latn"},
    "Yiddish": {"lang_code": "yi", "meta_code": "yi_Hebr"},
    "Yoruba": {"lang_code": "yo", "meta_code": "yo_Latn"},
    "Zulu": {"lang_code": "zu", "meta_code": "zul_Latn"},
}
available_language=['English','Hindi','Bengali','Akan', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Basque', 'Bashkir', 'Bengali', 'Bosnian', 'Bulgarian', 'Burmese', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Kurdish', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Nepali', 'Norwegian', 'Norwegian Nynorsk', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba', 'Zulu']
import math
import torch
import gc
import time
import subprocess
from faster_whisper import WhisperModel
import os
import mimetypes
import shutil
import re
import uuid
from pydub import AudioSegment


def get_language_name(lang_code):
    global language_dict
    # Iterate through the language dictionary
    for language, details in language_dict.items():
        # Check if the language code matches
        if details["lang_code"] == lang_code:
            return language  # Return the language name
    return None

def clean_file_name(file_path):
    # Get the base file name and extension
    file_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name)

    # Replace non-alphanumeric characters with an underscore
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)

    # Remove any multiple underscores
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')

    # Generate a random UUID for uniqueness
    random_uuid = uuid.uuid4().hex[:6]

    # Combine cleaned file name with the original extension
    clean_file_path = os.path.join(os.path.dirname(file_path), clean_file_name + f"_{random_uuid}" + file_extension)

    return clean_file_path

def get_audio_file(uploaded_file):
    global base_path
    # ,device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Detect the file type (audio/video)
    mime_type, _ = mimetypes.guess_type(uploaded_file)
    # Create the folder path to store audio files
    audio_folder = f"{base_path}/subtitle_audio"
    os.makedirs(audio_folder, exist_ok=True)
    # Initialize variable for the audio file path
    audio_file_path = ""
    if mime_type and mime_type.startswith('audio'):
        # If it's an audio file, save it as is
        audio_file_path = os.path.join(audio_folder, os.path.basename(uploaded_file))
        audio_file_path=clean_file_name(audio_file_path)
        shutil.copy(uploaded_file, audio_file_path)  # Move file to audio folder

    elif mime_type and mime_type.startswith('video'):
        # If it's a video file, extract the audio
        audio_file_name = os.path.splitext(os.path.basename(uploaded_file))[0] + ".mp3"
        audio_file_path = os.path.join(audio_folder, audio_file_name)
        audio_file_path=clean_file_name(audio_file_path)

        # Extract the file extension from the uploaded file
        file_extension = os.path.splitext(uploaded_file)[1]  # Includes the dot, e.g., '.mp4'

        # Generate a random UUID and create a new file name with the same extension
        random_uuid = uuid.uuid4().hex[:6]
        new_file_name = random_uuid + file_extension

        # Set the new file path in the subtitle_audio folder
        new_file_path = os.path.join(audio_folder, new_file_name)

        # Copy the original video file to the new location with the new name
        shutil.copy(uploaded_file, new_file_path)
        if device=="cuda":
          command = f"ffmpeg -hwaccel cuda -i {new_file_path} {audio_file_path} -y"
        else:
          command = f"ffmpeg -i {new_file_path} {audio_file_path} -y"

        subprocess.run(command, shell=True)
        if os.path.exists(new_file_path):
          os.remove(new_file_path)
    # Return the saved audio file path
    audio = AudioSegment.from_file(audio_file_path)
    # Get the duration in seconds
    duration_seconds = len(audio) / 1000.0  # pydub measures duration in milliseconds
    return audio_file_path,duration_seconds

def format_segments(segments):
    saved_segments = list(segments)
    sentence_timestamp = []
    words_timestamp = []
    speech_to_text = ""

    for i in saved_segments:
        temp_sentence_timestamp = {}
        # Store sentence information in sentence_timestamp
        text = i.text.strip()
        sentence_id = len(sentence_timestamp)  # Get the current index for the new entry
        sentence_timestamp.append({
            "id": sentence_id,  # Use the index as the id
            "text": text,
            "start": i.start,
            "end": i.end,
            "words": []  # Initialize words as an empty list within the sentence
        })
        speech_to_text += text + " "

        # Process each word in the sentence
        for word in i.words:
            word_data = {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            }

            # Append word timestamps to the sentence's word list
            sentence_timestamp[sentence_id]["words"].append(word_data)

            # Optionally, add the word data to the global words_timestamp list
            words_timestamp.append(word_data)

    return sentence_timestamp, words_timestamp, speech_to_text

def combine_word_segments(words_timestamp, max_words_per_subtitle=8, min_silence_between_words=0.5):
    before_translate = {}
    id = 1
    text = ""
    start = None
    end = None
    word_count = 0
    last_end_time = None

    for i in words_timestamp:
        try:
            word = i['word']
            word_start = i['start']
            word_end = i['end']

            # Check for sentence-ending punctuation
            is_end_of_sentence = word.endswith(('.', '?', '!'))

            # Check for conditions to create a new subtitle
            if ((last_end_time is not None and word_start - last_end_time > min_silence_between_words)
                or word_count >= max_words_per_subtitle
                or is_end_of_sentence):

                # Store the previous subtitle if there's any
                if text:
                    before_translate[id] = {
                        "text": text,
                        "start": start,
                        "end": end
                    }
                    id += 1

                # Reset for the new subtitle segment
                text = word
                start = word_start  # Set the start time for the new subtitle
                word_count = 1
            else:
                if word_count == 0:  # First word in the subtitle
                    start = word_start  # Ensure the start time is set
                text += " " + word
                word_count += 1

            end = word_end  # Update the end timestamp
            last_end_time = word_end  # Update the last end timestamp

        except KeyError as e:
            print(f"KeyError: {e} - Skipping word")
            pass

    # After the loop, make sure to add the last subtitle segment
    if text:
        before_translate[id] = {
            "text": text,
            "start": start,
            "end": end
        }

    return before_translate


def convert_time_to_srt_format(seconds):
    """ Convert seconds to SRT time format (HH:MM:SS,ms) """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
def write_subtitles_to_file(subtitles, filename="subtitles.srt"):

    # Open the file with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        for id, entry in subtitles.items():
            # Write the subtitle index
            f.write(f"{id}\n")
            if entry['start'] is None or entry['end'] is None:
              print(id)
            # Write the start and end time in SRT format
            start_time = convert_time_to_srt_format(entry['start'])
            end_time = convert_time_to_srt_format(entry['end'])
            f.write(f"{start_time} --> {end_time}\n")

            # Write the text and speaker information
            f.write(f"{entry['text']}\n\n")

def word_level_srt(words_timestamp, srt_path="world_level_subtitle.srt"):
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start_time = convert_time_to_srt_format(word_info['start'])
            end_time = convert_time_to_srt_format(word_info['end'])
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{word_info['word']}\n\n")

def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamp):
            start_time = convert_time_to_srt_format(sentence['start'])
            end_time = convert_time_to_srt_format(sentence['end'])
            srt_file.write(f"{index + 1}\n{start_time} --> {end_time}\n{sentence['text']}\n\n")



def whisper_subtitle(uploaded_file,Source_Language,max_words_per_subtitle=8):
  global language_dict,base_path
  #setup srt file names
  base_name = os.path.basename(uploaded_file).rsplit('.', 1)[0][:30]
  save_name = f"{base_path}/generated_subtitle/{base_name}_{Source_Language}.srt"
  original_srt_name=clean_file_name(save_name)
  original_txt_name=original_srt_name.replace(".srt",".txt")
  word_level_srt_name=original_srt_name.replace(".srt","_word_level.srt")
  default_srt_name=original_srt_name.replace(".srt","_default.srt")
  #Load model
  faster_whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2")
  audio_path,audio_duration=get_audio_file(uploaded_file)

  if Source_Language=="Automatic":
      segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True)
      lang_code=d.language
      src_lang=get_language_name(lang_code)
  else:
    lang=language_dict[Source_Language]['lang_code']
    segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True,language=lang)
    src_lang=Source_Language
  if os.path.exists(audio_path):
    os.remove(audio_path)


  sentence_timestamp,words_timestamp,text=format_segments(segments)
  del faster_whisper_model
  gc.collect()
  torch.cuda.empty_cache()

  word_segments=combine_word_segments(words_timestamp, max_words_per_subtitle=max_words_per_subtitle, min_silence_between_words=0.5)
  generate_srt_from_sentences(sentence_timestamp, srt_path=default_srt_name)
  word_level_srt(words_timestamp, srt_path=word_level_srt_name)
  write_subtitles_to_file(word_segments, filename=original_srt_name)
  with open(original_txt_name, 'w', encoding='utf-8') as f1:
    f1.write(text)
  return default_srt_name,original_srt_name,word_level_srt_name,original_txt_name

#@title Using Gradio Interface
def subtitle_maker(Audio_or_Video_File,Source_Language,max_words_per_subtitle):
  try:
    default_srt_path,customize_srt_path,word_level_srt_path,text_path=whisper_subtitle(Audio_or_Video_File,Source_Language,max_words_per_subtitle=max_words_per_subtitle)
  except:
    default_srt_path,customize_srt_path,word_level_srt_path,text_path=None,None,None,None
  return default_srt_path,customize_srt_path,word_level_srt_path,text_path


base_path="."
if not os.path.exists(f"{base_path}/generated_subtitle"):
    os.makedirs(f"{base_path}/generated_subtitle", exist_ok=True)
import gradio as gr
import click

source_lang_list = ['Automatic']
source_lang_list.extend(available_language)  # Ensure available_language is defined elsewhere


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    # Define Gradio inputs and outputs
    gradio_inputs = [
        gr.File(label="Upload Audio or Video File"),
        gr.Dropdown(label="Language", choices=source_lang_list, value="Automatic"),
        gr.Number(label="Max Word Per Subtitle Segment", value=8)
    ]
    
    gradio_outputs = [
        gr.File(label="Default SRT File", show_label=True),
        gr.File(label="Customize Text File", show_label=True),
        gr.File(label="Word Level SRT File", show_label=True),
        gr.File(label="Text File", show_label=True)
    ]

    # Create Gradio interface
    demo = gr.Interface(fn=subtitle_maker, inputs=gradio_inputs, outputs=gradio_outputs, title="Whisper-Large-V3-Turbo-Ct2 Subtitle Maker")

    # Launch Gradio with command-line options
    demo.launch(debug=debug, share=share)

if __name__ == "__main__":
    main()
