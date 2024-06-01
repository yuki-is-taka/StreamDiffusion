import os
import gradio as gr
import sys
from utils.wrapper import StreamDiffusionWrapper
import time

def list_files_in_folder(hf_home):
    if hf_home:
        hf_home = f'{hf_home}/hub'
        folders = [folder.replace('models--', '').replace('--', '/') for folder in os.listdir(hf_home) if os.path.isdir(os.path.join(hf_home, folder))]
        return folders

def stream_engine(width, height, steps, acceleration, model, provided_model):

    model_id_or_path = provided_model if provided_model else model

    # Determine acceleration method
    use_lcm_lora = acceleration == 'LCM'
    turbo = acceleration == 'Turbo'
    # Generate time index list
    t_index_list = list(range(steps))
    

    engine_dir = f'{parent_dir}/engines'

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=None,
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=0,
        acceleration="tensorrt",
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="none",
        seed=2,
        use_lcm_lora = use_lcm_lora,
        touchdiffusion = False,
        engine_dir=engine_dir,
        turbo=turbo
    )

    stream.prepare(
        prompt="1girl with brown dog hair, thick glasses, smiling",
        negative_prompt="low quality, bad quality, blurry, low resolution",
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=0.5,
        t_index_list=t_index_list
    )

    input = f'{parent_dir}/StreamDiffusion/images/inputs/input.png'
    #os.path.abspath('StreamDiffusion/images/inputs/input.png')
    image_tensor = stream.preprocess_image(input)
    fps = 0
    for i in range(10):
        start_time = time.time()
        last_element = 1 if stream.batch_size != 1 else 0
        for _ in range(stream.batch_size - last_element):
            stream(image=image_tensor)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time != 0:
            fps = 1 / elapsed_time

    return f"""Model: {model_id_or_path}\nWxH: {width}x{height}\nBatch size: {steps}\nExpected: {int(fps)} FPS\nStatus: Ready"""


current_directory = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_directory, os.pardir))
hf_home = os.getenv('HF_HOME')
if not hf_home:
    os.environ['HF_HOME'] = f'{parent_dir}/models'
    
models = list_files_in_folder(hf_home)

demo = gr.Interface(
    stream_engine,
    [   gr.Slider(512, 512, value=512, step = 2, label='Width'),
        gr.Slider(512, 512, value=512, step = 2, label='Height'),
        gr.Slider(1, 20, value=1, step = 1, label='Sampling steps (Batch size)'),
        gr.Radio(["None", "Turbo", "LCM"], label='Acceleration'),
        gr.Dropdown(models, label=f"Select model from {hf_home}"),
        gr.Textbox(label=f"Or provide model name")
     
    ],
    "text",
    allow_flagging='never')

if __name__ == "__main__":
    demo.launch()