# https://huggingface.co/hexgrad/Kokoro-82M
# voices : https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md#japanese

from kokoro import KPipeline
import soundfile as sf
import torch
import json
import os
import time

todo_dir = "/app/tmp/todo"
done_dir = "/app/tmp/done"

print("Load pipe")
pipeline = KPipeline(lang_code='j')
print("pipe loaded")


while True:
    files = [f for f in os.listdir(todo_dir) if f.endswith(".json")]

    if files:
        for filename in files:
            filepath = os.path.join(todo_dir, filename)
            try:
                print(f"Processing {filename}...")
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text = data.get("text")
                outfile = done_dir + "/" + data.get("filename", "output") + ".wav"
                voice = data.get("voice", "jf_alpha")

                generator = pipeline(text, voice=voice)
                for i, (gs, ps, audio) in enumerate(generator):
                    sf.write(outfile, audio, 24000)
                    data = {
                        "i": i,
                        "gs": gs,
                        "ps": ps,
                        "voice": voice,
                        "text": text
                    }
                    with open(f'{i}.json', 'w') as f:
                        json.dump(data, f, indent=4)

                os.remove(filepath)

            except Exception as e:
                print(f"[ERROR] failed to process {filename} : {e}")

    time.sleep(2)
