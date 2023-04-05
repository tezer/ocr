import logging
import sys
from typing import List

import click
import numpy as np

import _io
import cv2
from preprocessor import process_image
from pdf2image import convert_from_path

try:
    from PIL import Image
except ImportError:
    import Image

dev_logger_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
user_logger_format = f"%(asctime)s - %(message)s"


def get_file_handler():
    file_handler = logging.FileHandler("dev.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(dev_logger_format))
    return file_handler


def get_stream_handler():
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(user_logger_format))
    return stream_handler


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler())
    logger.addHandler(get_stream_handler())
    return logger


logger = get_logger(__name__)


def do_pdf_ocr(input_file: _io.BufferedReader, language: str) -> str:
    images = convert_from_path(input_file.name)
    logger.info(f"Processing a PDF file, containing {len(images)} pages")
    texts: List[str] = []
    for image in images:
        logger.info(f"Processing page {len(texts) + 1}")
        im = np.asarray(image)
        text = process_image(im, language=language)
        texts.append(text)
        logger.info(
            f"Finished processing page {len(texts)} out of {len(images)}")
    document = '\n\n'.join(texts)
    return document


@click.command()
@click.option('--input_file',
              type=click.File("rb"),
              prompt='Specify the input file',
              help='Enter the path to the file to be processed.')
@click.option(
    '--output_file',
    type=click.File("w"),
    default="output.txt",
    prompt='Specify the output file (default is output.txt)',
    help=
    'You need to write down a file name where the output will be saved. The default value is output.txt'
)
@click.option(
    '--language',
    default="en",
    prompt='Specify the language of the input file (default is en)',
)
@click.option(
    '--verbose',
    default=False,
    prompt='Output more information about processing',
    help=
    'If you are bored and like reading, use this option for the command to wordy describing what it is '
    'doing.')
def do_ocr(input_file: _io.BufferedReader, output_file: _io.TextIOWrapper,
           verbose: bool, language: str) -> None:
    click.echo(
        f"Input file is {input_file.name}, output file is {output_file.name}, and verbose is {verbose}"
    )
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    if input_file.name.endswith('.pdf'):
        text = do_pdf_ocr(input_file, language=language)
    else:
        im = cv2.imread(input_file.name)
        text = process_image(im, language=language)
    logger.info(
        f"Finished image recognition, writing out result into {output_file.name}"
    )
    output_file.write(text)


if __name__ == '__main__':
    do_ocr()
