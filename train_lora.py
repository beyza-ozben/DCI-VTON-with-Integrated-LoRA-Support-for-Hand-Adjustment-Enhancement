import subprocess
import os

# Correct paths
kohya_dir = os.path.expanduser("~/Developer/sga/kohya_ss")
data_dir = os.path.expanduser("~/Developer/sga/DCI-VTON-Virtual-Try-On/data")

cmd = [
    os.path.join(kohya_dir, "venv", "bin", "python"),
    os.path.join(kohya_dir, "train_network.py"),
    "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
    f"--train_data_dir={data_dir}",
    f"--dataset_config={os.path.join(data_dir, 'prompts.json')}",
    f"--output_dir={os.path.join(os.path.dirname(data_dir), 'lora_output')}",
    "--network_dim=64",
    "--network_alpha=32",
    "--batch_size=4",
    "--max_train_steps=1500",
    "--mixed_precision=fp16",
    "--xformers"
]

subprocess.run(cmd, check=True)
