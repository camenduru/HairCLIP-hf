#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess

import gradio as gr

if os.getenv('SYSTEM') == 'spaces':
    with open('patch.e4e') as f:
        subprocess.call('patch -p1'.split(), cwd='encoder4editing', stdin=f)
    with open('patch.hairclip') as f:
        subprocess.call('patch -p1'.split(), cwd='HairCLIP', stdin=f)

from model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


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


def main():
    args = parse_args()
    model = Model(device=args.device)

    css = '''
h1#title {
  text-align: center;
}
img#teaser {
  max-width: 1000px;
  max-height: 600px;
}
'''

    with gr.Blocks(theme=args.theme, css=css) as demo:
        gr.Markdown('''<h1 id="title">HairCLIP</h1>

This is an unofficial demo for <a href="https://github.com/wty-ustc/HairCLIP">https://github.com/wty-ustc/HairCLIP</a>.

<center><img id="teaser" src="https://raw.githubusercontent.com/wty-ustc/HairCLIP/main/assets/teaser.png" alt="teaser"></center>
''')
        with gr.Box():
            gr.Markdown('## Step 1')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type='file')
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
                example_images = gr.Dataset(components=[input_image],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('## Step 2')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        editing_type = gr.Radio(['hairstyle', 'color', 'both'],
                                                value='both',
                                                type='value',
                                                label='Editing Type')
                    with gr.Row():
                        hairstyles = load_hairstyle_list()
                        hairstyle_index = gr.Dropdown(hairstyles,
                                                      value='afro',
                                                      type='index',
                                                      label='Hairstyle')
                    with gr.Row():
                        color_description = gr.Textbox(value='red',
                                                       label='Color')
                    with gr.Row():
                        run_button = gr.Button('Run')

                with gr.Column():
                    result = gr.Image(label='Result')

        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.hairclip" alt="visitor badge"/></center>'
        )

        preprocess_button.click(fn=model.detect_and_align_face,
                                inputs=[input_image],
                                outputs=[aligned_face])
        aligned_face.change(fn=model.reconstruct_face,
                            inputs=[aligned_face],
                            outputs=[reconstructed_face, latent])
        editing_type.change(fn=update_step2_components,
                            inputs=[editing_type],
                            outputs=[hairstyle_index, color_description])
        run_button.click(fn=model.generate,
                         inputs=[
                             editing_type,
                             hairstyle_index,
                             color_description,
                             latent,
                         ],
                         outputs=[result])
        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
