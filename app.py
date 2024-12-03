from utils import language_dict
import math
import torch
import gc
import time
from faster_whisper import WhisperModel
import os
import re
import uuid
import shutil


def get_language_name(lang_code):
    global language_dict
    # Iterate through the language dictionary
    for language, details in language_dict.items():
        # Check if the language code matches
        if details["lang_code"] == lang_code:
            return language  # Return the language name
    return lang_code

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
    if max_words_per_subtitle<=1:
        max_words_per_subtitle=1
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

def custom_word_segments(words_timestamp, min_silence_between_words=0.3, max_characters_per_subtitle=17):
    before_translate = []
    id = 1
    text = ""
    start = None
    end = None
    last_end_time = None

    i = 0
    while i < len(words_timestamp):
        word = words_timestamp[i]['word']
        word_start = words_timestamp[i]['start']
        word_end = words_timestamp[i]['end']

        # Look ahead to check if the next word (i+1) starts with a hyphen
        if i + 1 < len(words_timestamp) and words_timestamp[i + 1]['word'].startswith("-"):
            # Combine the current word and the next word (i, i+1) if next word starts with a hyphen
            combined_text = word + words_timestamp[i + 1]['word'][:]  # Skip the hyphen and combine
            combined_start_time = word_start
            combined_end_time = words_timestamp[i + 1]['end']

            i += 1  # Skip the next word (i+1) since it has been combined

            # Look ahead for the next non-hyphenated word, check further if needed (i+2, i+3, etc.)
            while i + 1 < len(words_timestamp) and words_timestamp[i + 1]['word'].startswith("-"):
                combined_text += words_timestamp[i + 1]['word'][:]  # Add word excluding hyphen
                combined_end_time = words_timestamp[i + 1]['end']
                i += 1  # Skip the next hyphenated word

        else:
            # No hyphen at the next word, just take the current word
            combined_text = word
            combined_start_time = word_start
            combined_end_time = word_end

        # Check if the combined text exceeds the maximum character limit
        if len(text) + len(combined_text) > max_characters_per_subtitle:
            # If accumulated text is non-empty, store it as a subtitle
            if text:
                before_translate.append({
                    "word": text.strip(),
                    "start": start,
                    "end": end
                })
                id += 1
            # Start a new subtitle with the combined text
            text = combined_text
            start = combined_start_time
        else:
            # Accumulate text
            if not text:
                start = combined_start_time
            text += " " + combined_text

        # Update the end timestamp
        end = combined_end_time
        last_end_time = end

        # Move to the next word
        i += 1

    # Add the final subtitle segment if text is not empty
    if text:
        before_translate.append({
            "word": text.strip(),
            "start": start,
            "end": end
        })

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


def word_level_srt(words_timestamp, srt_path="world_level_subtitle.srt",shorts=False):
    punctuation_pattern = re.compile(r'[.,!?;:"\–—_~^+*|]')
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start_time = convert_time_to_srt_format(word_info['start'])
            end_time = convert_time_to_srt_format(word_info['end'])
            word=word_info['word']
            word =re.sub(punctuation_pattern, '', word)
            if word.strip() == 'i':
                word = "I"
            if shorts==False:
              word=word.replace("-","")
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{word}\n\n")


def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamp):
            start_time = convert_time_to_srt_format(sentence['start'])
            end_time = convert_time_to_srt_format(sentence['end'])
            srt_file.write(f"{index + 1}\n{start_time} --> {end_time}\n{sentence['text']}\n\n")

def get_audio_file(uploaded_file):
    global temp_folder
    file_path = os.path.join(temp_folder, os.path.basename(uploaded_file))
    file_path=clean_file_name(file_path)
    shutil.copy(uploaded_file, file_path)
    return file_path

def whisper_subtitle(uploaded_file,Source_Language,max_words_per_subtitle=8):
  global language_dict,base_path,subtitle_folder
  #Load model
  if torch.cuda.is_available():
      # If CUDA is available, use GPU with float16 precision
      device = "cuda"
      compute_type = "float16"
      # compute_type="int8_float16"
  else:
      # If CUDA is not available, use CPU with int8 precision
      device = "cpu"
      compute_type = "int8"
  faster_whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2",device=device, compute_type=compute_type)
  audio_path=get_audio_file(uploaded_file)
  if Source_Language=="Automatic":
      segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True)
      lang_code=d.language
      src_lang=get_language_name(lang_code)
  else:
    lang=language_dict[Source_Language]['lang_code']
    segments,d = faster_whisper_model.transcribe(audio_path, word_timestamps=True,language=lang)
    src_lang=Source_Language
      
  sentence_timestamp,words_timestamp,text=format_segments(segments)
  if os.path.exists(audio_path):
    os.remove(audio_path)
  del faster_whisper_model
  gc.collect()
  torch.cuda.empty_cache()
  
  word_segments=combine_word_segments(words_timestamp, max_words_per_subtitle=max_words_per_subtitle, min_silence_between_words=0.5)
  shorts_segments=custom_word_segments(words_timestamp, min_silence_between_words=0.3, max_characters_per_subtitle=17)
  #setup srt file names
  base_name = os.path.basename(uploaded_file).rsplit('.', 1)[0][:30]
  save_name = f"{subtitle_folder}/{base_name}_{src_lang}.srt"
  original_srt_name=clean_file_name(save_name)
  original_txt_name=original_srt_name.replace(".srt",".txt")
  word_level_srt_name=original_srt_name.replace(".srt","_word_level.srt")
  customize_srt_name=original_srt_name.replace(".srt","_customize.srt")
  shorts_srt_name=original_srt_name.replace(".srt","_shorts.srt")
    
  generate_srt_from_sentences(sentence_timestamp, srt_path=original_srt_name)
  word_level_srt(words_timestamp, srt_path=word_level_srt_name)
  word_level_srt(shorts_segments, srt_path=shorts_srt_name,shorts=True)
  write_subtitles_to_file(word_segments, filename=customize_srt_name)
  with open(original_txt_name, 'w', encoding='utf-8') as f1:
    f1.write(text)
  return original_srt_name,customize_srt_name,word_level_srt_name,shorts_srt_name,original_txt_name

#@title Using Gradio Interface
def subtitle_maker(Audio_or_Video_File,Source_Language,max_words_per_subtitle):
  try:
    default_srt_path,customize_srt_path,word_level_srt_path,shorts_srt_name,text_path=whisper_subtitle(Audio_or_Video_File,Source_Language,max_words_per_subtitle=max_words_per_subtitle)
  except Exception as e:
    print(f"Error in whisper_subtitle: {e}")
    default_srt_path,customize_srt_path,word_level_srt_path,shorts_srt_name,text_path=None,None,None,None,None
  return default_srt_path,customize_srt_path,word_level_srt_path,shorts_srt_name,text_path





import gradio as gr
import click

base_path="."
subtitle_folder=f"{base_path}/generated_subtitle"
temp_folder = f"{base_path}/subtitle_audio"

if not os.path.exists(subtitle_folder):
    os.makedirs(subtitle_folder, exist_ok=True)
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder, exist_ok=True)
    
source_lang_list = ['Automatic']
available_language=language_dict.keys()
source_lang_list.extend(available_language)  



@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
    description = """**Note**: Avoid uploading large video files. Instead, upload the audio from the video for faster processing.
    You can find the model at [faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)"""
    # Define Gradio inputs and outputs
    gradio_inputs = [
        gr.File(label="Upload Audio or Video File"),
        gr.Dropdown(label="Language", choices=source_lang_list, value="Automatic"),
        gr.Number(label="Max Word Per Subtitle Segment [Useful for Vertical Videos]", value=8)
    ]
    
    gradio_outputs = [
        gr.File(label="Default SRT File", show_label=True),
        gr.File(label="Customize SRT File", show_label=True),
        gr.File(label="Word Level SRT File", show_label=True),
        gr.File(label="SRT File For Shorts", show_label=True),
        gr.File(label="Text File", show_label=True)
    ]

    # Create Gradio interface
    demo = gr.Interface(fn=subtitle_maker, inputs=gradio_inputs, outputs=gradio_outputs, title="Auto Subtitle Generator Using Whisper-Large-V3-Turbo-Ct2",description=description)

    # Launch Gradio with command-line options
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()
