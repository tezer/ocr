# ocr

This is a simple command-line OCR utility for extracting text from image files (including multipage PDF images).

## Installation
clone the project with `git clone https://github.com/tezer/ocr.git`

## Usage
Requires python 3.8
1. `cd ocr`

2. `poetry run python nexus_test/rec.py` for interactive mode

or 

```
poetry run python rec.py --input=./test.pdf --output=output.text --verbose
```
where
```
Options:
  --input_file FILENAME   Enter the path to the file to be processed.
  --output_file FILENAME  You need to write down a file name where the output
                          will be saved. The default value is output.txt
  --verbose BOOLEAN       If you are bored and like reading, use this option
                          for the command to wordy describing what it is
                          doing.
  --help                  Show this message and exit.
```