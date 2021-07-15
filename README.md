# ocr

This is a simple command-line OCR utility for extracting text from image files (including multipage PDF images).

## Installation
1. clone the project
1. install dependencies with `poetry install`

## Usage
Requires python 3.8
```
python rec.py --input=./test.pdf --output=output.text --verbose
–input : input image ﬁle
–output : output text ﬁle
–verbose : verbose mode ( output detailed logs )
```