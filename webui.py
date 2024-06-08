import os
import gradio as gr
import sys
import time
import subprocess
import platform

#import torch

try:
    from utils.wrapper import StreamDiffusionWrapper
except ImportError:
    print('Dependencies not installed')

def list_files_in_folder(hf_home):
    if hf_home:
        hf_home = f'{hf_home}/hub'
        folders = [folder.replace('models--', '').replace('--', '/') for folder in os.listdir(hf_home) if os.path.isdir(os.path.join(hf_home, folder))]
        return folders

# def download_model(model_url):
#     pipeline = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo", varia)
#     # hf_hub_download(
#     #     v
#     # )

#     # snapshot_download(
#     #     repo_id=model_url,
#     #     cache_dir='../models/diffusers',
#     #     revision='fp16'        
#     #     )

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

            print(fps)

    return f"""Model: {model_id_or_path}\nWxH: {width}x{height}\nBatch size: {steps}\nExpected: {int(fps)} FPS\nStatus: Ready"""

def is_installed(package_name):
    try:
        subprocess.run(["pip", "show", package_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def inst_upd():
    cu="11"
    error_packages = []
    with open('requirements.txt', "r") as file:
        packages = file.read().splitlines()

    try:
        subprocess.run(["git", "pull"], check=True)
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.check_call(["pip", "install", "torch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2", "--index-url", "https://download.pytorch.org/whl/cu118"])
    except Exception as e:
        print(f"An unexpected error occurred while executing the command: {e}")
        error_packages.append('torch==2.2.2')

    for package in packages:
        try:
            subprocess.check_call(["pip", "install", package])
        except subprocess.CalledProcessError:
            error_packages.append(package)
        except Exception as e:
            print(f"An unexpected error occurred while installing {package}: {e}")
            error_packages.append(package)

    if cu is None or cu not in ["11", "12"]:
        print("Could not detect CUDA version. Please specify manually.")
        error_packages.append("CUDA version not detected")
        return

    print("Installing TensorRT requirements...")

    cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"

    # Install TensorRT
    if not is_installed("tensorrt"):
        try:
            subprocess.run(["pip", "install", f"{cudnn_name}", "--no-cache-dir"], check=True)
            subprocess.run(["pip", "install", "--pre", "--extra-index-url", "https://pypi.nvidia.com", "tensorrt==9.0.1.post11.dev4", "--no-cache-dir"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install TensorRT")

    # Install other required packages
    if not is_installed("polygraphy"):
        try:
            subprocess.run(["pip", "install", "polygraphy==0.47.1", "--extra-index-url", "https://pypi.ngc.nvidia.com"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install polygraphy")
    if not is_installed("onnx_graphsurgeon"):
        try:
            subprocess.run(["pip", "install", "onnx-graphsurgeon==0.3.26", "--extra-index-url", "https://pypi.ngc.nvidia.com"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install onnx-graphsurgeon")
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        try:
            subprocess.run(["pip", "install", "pywin32"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install pywin32")

    if error_packages:
        return f"Error installing packages: {', '.join(error_packages)}"
    else:
        return "All packages installed successfully! You need to restart webui.bat to apply changes."


current_directory = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_directory, os.pardir))
hf_home = os.getenv('HF_HOME')
if not hf_home:
    os.environ['HF_HOME'] = f'{parent_dir}/models'
    
models = list_files_in_folder(hf_home)

with gr.Blocks() as demo:
    with gr.Tab("Engine"):
        with gr.Row():
            with gr.Column(scale=1):
                width_slider = gr.Slider(512, 512, value=512, step=2, label='Width', interactive=True)
                height_slider = gr.Slider(512, 512, value=512, step=2, label='Height', interactive=True)
                sampling_steps_slider = gr.Slider(1, 20, value=1, step=1, label='Sampling steps (Batch size)', interactive=True)
                acceleration_radio = gr.Radio(["None", "Turbo", "LCM"], label='Acceleration')
                model_dropdown = gr.Dropdown(models, label=f"Select model from {hf_home}")
                model_textbox = gr.Textbox(label=f"Or provide model name")
            with gr.Column(scale=1):
                output = gr.Textbox(label="Output")
                make_engine = gr.Button("Make engine")

            make_engine.click(fn=stream_engine, 
                              inputs=[width_slider, height_slider,
                                      sampling_steps_slider, acceleration_radio,
                                      model_dropdown, model_textbox], 
                              outputs=output)

    # with gr.Tab("Download model"):
    #     with gr.Row():
    #         with gr.Column(scale=1):
    #             # Button for triggering update
    #             gr.Text("This action will download model weights.", label='Description')
    #             model_url = gr.Textbox(label=f"Model name (ex. stabilityai/sd-turbo)")
    #             download = gr.Button("Download")

    #             # Textbox to display output
    #         with gr.Column(scale=1):
    #             output = gr.Textbox(label="Output")
    #             # Attach click event to the button to trigger inst_upd function
                
    #         download.click(fn=download_model, inputs=[model_url], outputs=output)
    with gr.Tab("Install & Update"):
        with gr.Row():
            with gr.Column(scale=1):
                # Button for triggering update
                gr.Text("This action will install or update required dependencies.", label='Description')
                install_update = gr.Button("Update dependencies")
                # Textbox to display output
            with gr.Column(scale=1):
                output = gr.Textbox(label="Output")
                # Attach click event to the button to trigger inst_upd function

            install_update.click(fn=inst_upd, inputs=[], outputs=output)

if __name__ == "__main__":
    demo.launch()