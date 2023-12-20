import sys
import os
from email.parser import BytesParser
from email import policy
import html2text
import re
import summarize

input_dir = sys.argv[1]
output_dir = sys.argv[2]

summary = len(sys.argv) >= 4 and sys.argv[3] == '-s'

parser = BytesParser(policy=policy.default)

index = 0
for current_dir, subdirs, files in os.walk(input_dir):
    for filename in files:
        full_filename = os.path.join(current_dir, filename)
        with open(full_filename, "rb") as f:
            msg = parser.parse(f)
        text = ""
        if len(msg.keys()):
            text = f"{msg["Subject"]} " 
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    text += part.as_string()
                elif content_type == 'text/html':
                    try:
                        text += html2text.html2text(part.as_string())
                    except AssertionError:
                        continue
        words = [x for x in re.findall(r'\b\w+\b', text) if len(x) <= 15]
        text = ' '.join(words)
        if len(text) > 0:
            if summary:
                text = summarize.text_summarization(text)
                if not text:
                    print(f"{full_filename} is invalid or is too long to summarize")
                    continue
            output_file = os.path.join(output_dir, f"{index}.txt")
            with open(output_file, "w") as f:
                f.write(text)
            index += 1