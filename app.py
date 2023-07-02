#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from model import Model

DESCRIPTION = '''# [HairCLIP](https://github.com/wty-ustc/HairCLIP)

<center><img id="teaser" src="https://raw.githubusercontent.com/wty-ustc/HairCLIP/main/assets/teaser.png" alt="teaser"></center>
'''


def load_hairstyle_list() -> list[str]:
    with open('HairCLIP/mapper/hairstyle_list.txt') as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [line[:-10] for line in lines]
    return lines


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def update_step2_components(choice: str) -> tuple[dict, dict]:
    return (
        gr.Dropdown.update(visible=choice in ['hairstyle', 'both']),
        gr.Textbox.update(visible=choice in ['color', 'both']),
    )


model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Box():
        gr.Markdown('## Step 1')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image',
                                           type='filepath')
                with gr.Row():
                    preprocess_button = gr.Button('Preprocess')
            with gr.Column():
                aligned_face = gr.Image(label='Aligned Face',
                                        type='pil',
                                        interactive=False)
            with gr.Column():
                reconstructed_face = gr.Image(label='Reconstructed Face',
                                              type='numpy')
                latent = gr.Variable()

        with gr.Row():
            paths = sorted(pathlib.Path('images').glob('*.jpg'))
            gr.Examples(examples=[[path.as_posix()] for path in paths],
                        inputs=input_image)

    with gr.Box():
        gr.Markdown('## Step 2')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    editing_type = gr.Radio(
                        label='Editing Type',
                        choices=['hairstyle', 'color', 'both'],
                        value='both',
                        type='value')
                with gr.Row():
                    hairstyles = load_hairstyle_list()
                    hairstyle_index = gr.Dropdown(label='Hairstyle',
                                                  choices=hairstyles,
                                                  value='afro',
                                                  type='index')
                with gr.Row():
                    color_description = gr.Textbox(label='Color', value='red')
                with gr.Row():
                    run_button = gr.Button('Run')

            with gr.Column():
                result = gr.Image(label='Result')

    preprocess_button.click(fn=model.detect_and_align_face,
                            inputs=input_image,
                            outputs=aligned_face)
    aligned_face.change(fn=model.reconstruct_face,
                        inputs=aligned_face,
                        outputs=[reconstructed_face, latent])
    editing_type.change(fn=update_step2_components,
                        inputs=editing_type,
                        outputs=[hairstyle_index, color_description])
    run_button.click(fn=model.generate,
                     inputs=[
                         editing_type,
                         hairstyle_index,
                         color_description,
                         latent,
                     ],
                     outputs=result)

demo.queue(max_size=10).launch()
