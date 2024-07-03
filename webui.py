import os
import gradio as gr
import sys
import time
import subprocess
import platform
from pathlib import Path
import requests
import json
import webbrowser
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
git_executable = os.path.join(current_dir, '..', 'PortableGit', 'cmd', 'git.exe')
print(git_executable)

try:
    from utils.wrapper import StreamDiffusionWrapper
except ImportError:
    print('Dependencies not installed')

def list_files_in_folder(folder_path):
    directories = ['models', 'models/acceleration_loras', 'models/checkpoints', 'models/loras', 'models/vae', 'engines']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

    file_list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    return file_list

def stream_engine(width, height, steps, acceleration, model_id_or_path, model_type):

    use_lcm_lora = acceleration == 'LCM'
    use_hyper_lora = acceleration == 'HyperSD'

    t_index_list = list(range(steps))
    
    engine_dir = f'engines'
    model_id_or_path = os.path.splitext(model_id_or_path)[0]
    print(model_id_or_path)

    if model_id_or_path == 'SD Turbo':
        model_id_or_path = 'sd-turbo'

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
        cfg_type="self",
        seed=2,
        use_lcm_lora = use_lcm_lora,
        use_hyper_lora = use_hyper_lora,
        touchdiffusion = False,
        engine_dir=engine_dir,
        model_type=model_type
    )

    stream.prepare(
        prompt="1girl with brown dog hair, thick glasses, smiling",
        negative_prompt="low quality, bad quality, blurry, low resolution",
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=0.5,
        t_index_list=t_index_list
    )
    
    input = f'{current_dir}/images/inputs/input.png'

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

def git_fn(git_type):
    repo_url = 'https://github.com/olegchomp/StreamDiffusion'

    try:
        subprocess.run(["git", git_type, repo_url], check=True)
    except FileNotFoundError:
        print("Git is not installed or not found. Using portable Git instead.")
        subprocess.run([git_executable, git_type, repo_url], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to {git_type} the repository.")
    
    return

def is_installed(package_name):
    try:
        subprocess.run([sys.executable, "-m", "pip", "show", package_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    
def update_acceleration_lora(amount_steps, model_type):
        if model_type != 'sd_1.5_turbo' and model_type != None:
            if amount_steps in [1,2,4,8]:
                return ["None", "LCM", 'HyperSD']
            else:
                return ["None", "LCM"]

def update_interactivity(selected_value, amount_steps):
    if selected_value == "sd_1.5_turbo":
        return gr.Dropdown(value="SD Turbo", allow_custom_value=True, interactive=False), gr.Radio(["None"], value='None', label='Acceleration Lora not available for this model')
    else:
        if len(models) > 0:
            value = models[0]
        else:
            value = ''
        lora_list = update_acceleration_lora(amount_steps, selected_value)
        return gr.Dropdown(models, value=value, interactive=True), gr.Radio(lora_list, value='None', label='Add Acceleration Lora')

def update_hypersd(amount_steps, model_type):
    lora_list = update_acceleration_lora(amount_steps, model_type)
    return gr.Radio(lora_list, value='None', label='Add Acceleration Lora')

def inst_upd():
    cu="11"
    error_packages = []

    os.chdir(current_dir)
    
    git_fn("pull")

    with open(f'requirements.txt', "r") as file:
        packages = file.read().splitlines()

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
    except Exception as e:
        print(f"An unexpected error occurred while executing the command: {e}")
        error_packages.append('torch==2.1.0')

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
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
            subprocess.run([sys.executable, "-m", "pip", "install", f"{cudnn_name}", "--no-cache-dir"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--pre", "--extra-index-url", "https://pypi.nvidia.com", "tensorrt==9.0.1.post11.dev4", "--no-cache-dir"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install TensorRT")

    # Install other required packages
    if not is_installed("polygraphy"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "polygraphy==0.47.1", "--extra-index-url", "https://pypi.ngc.nvidia.com"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install polygraphy")
    if not is_installed("onnx_graphsurgeon"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "onnx-graphsurgeon==0.3.26", "--extra-index-url", "https://pypi.ngc.nvidia.com"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install onnx-graphsurgeon")
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"], check=True)
        except subprocess.CalledProcessError:
            error_packages.append("Failed to install pywin32")
    try:
        import torch
        import tensorrt as trt
        print(trt)
        print("All packages installed successfully! You need to restart webui.bat to apply changes.")
    except Exception as e:
        print(f"An unexpected error occurred while executing the command: {e}")

    if error_packages:
        return f"Error installing packages: {', '.join(error_packages)}"
    else:
        return "All packages installed successfully! You need to restart webui.bat to apply changes."

def fix_pop():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"])
        return 'Done. You need to restart webui.bat to apply changes.'
    except Exception as e:
        return f"An unexpected error occurred while executing the command: {e}"

       
def check_version():
    url = 'https://api.github.com/repos/olegchomp/TouchDiffusion/releases/latest'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            gitversion = data['name']
            return gitversion
        else:
            return 'No internet connection'
    except Exception as e:
        return 'No internet connection'

def open_link(link):
    webbrowser.open_new_tab(link)

current_directory = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_directory, os.pardir))
print(current_dir, parent_dir)

models = list_files_in_folder('models/checkpoints')
model_type = ['sd_1.5', 'sd_1.5_turbo']

button_info = [
    ("Github", "https://github.com/olegchomp/TouchDiffusion"),
    ("Discord", "https://discord.gg/wNW8xkEjrf"),
    ("Youtube", "https://www.youtube.com/vjschool"),
    ("Telegram", "https://t.me/vjschool"),
    ("Author", "https://olegcho.mp/"),
    ("Buy me a coffee", "https://boosty.to/vjschool/")
    ]

with gr.Blocks() as demo:
    with gr.Tab("Engine"):
        with gr.Row():
            with gr.Column(scale=1):
                model_type = gr.Dropdown(model_type, label=f"Select model type")
                model_dropdown = gr.Dropdown(models, label=f"Select model", scale=2)
                width_slider = gr.Slider(256, 1024, value=512, step=8, label='Width', interactive=True)
                height_slider = gr.Slider(256, 1024, value=512, step=8, label='Height', interactive=True)
                sampling_steps_slider = gr.Slider(1, 20, value=1, step=1, label='Sampling steps (Batch size)', interactive=True)
                acceleration_radio = gr.Radio(["None"], value='None', label='Add Acceleration Lora')
            with gr.Column(scale=1):
                output = gr.Textbox(label="Output")
                make_engine = gr.Button("Make engine", variant='primary')

            make_engine.click(fn=stream_engine, 
                              inputs=[width_slider, height_slider,
                                      sampling_steps_slider, acceleration_radio,
                                      model_dropdown, model_type], 
                              outputs=output)
            model_type.change(fn=update_interactivity, inputs=[model_type,sampling_steps_slider], outputs=[model_dropdown, acceleration_radio])
            sampling_steps_slider.change(fn=update_hypersd, inputs=[sampling_steps_slider,model_type], outputs=[acceleration_radio])
    
    with gr.Tab("Install & Update"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Text("This action will install or update required dependencies.", label='Description')
                install_update = gr.Button("Update dependencies", variant='primary')
                gr.Text("If you get pop up window with error, click 'fix pop up' button.", label='Additional step')
                fix_popup = gr.Button("Fix pop up", variant='secondary')
            with gr.Column(scale=1):
                output = gr.Textbox(label="Output")

            install_update.click(fn=inst_upd, inputs=[], outputs=output)
            fix_popup.click(fn=fix_pop, inputs=[], outputs=output)
    with gr.Tab("About"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Text("TouchDiffusion-v.0.0.3", label='Your version', interactive=False)
                gr.Text(check_version, label='Latest version', interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    for label, url in button_info:
                        button = gr.Button(label, variant='secondary')
                        button.click(lambda url=url: open_link(url))
                        
if __name__ == "__main__":
    demo.launch()
