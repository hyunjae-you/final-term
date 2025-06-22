import os
import subprocess
import random
import shutil

# Hyperparameter search spaces
lrs = [0.001, 0.0005, 0.0001]
batch_sizes = [32, 64, 128]
n_convs = [2, 3, 4]
h_fea_lens = [64, 128, 256]

# Settings for faster execution
num_trials = 5 # Reduce number of trials for quicker testing
epochs_per_trial = 10 # Reduce epochs per trial for quicker testing (most impactful)
num_data_workers = 4 # Increase data loading workers to mitigate CPU bottleneck

data_dir_for_main_py = "/home/edgpu/edgpu27/cgcnn_test/hands_on/1/HexOx_cifs" # Ensure this path is correct

for i in range(num_trials):
    lr = random.choice(lrs)
    batch_size = random.choice(batch_sizes)
    n_conv = random.choice(n_convs)
    h_fea_len = random.choice(h_fea_lens)

    print(f"--- Trial {i+1}/{num_trials} ---")
    print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, n_conv={n_conv}, h_fea_len={h_fea_len}")
    print(f"Epochs: {epochs_per_trial}, Data Workers: {num_data_workers}")

    # Define output log directory for this trial
    trial_output_dir = f"trial_results_fast_test/trial_{i+1}_lr{lr}_bs{batch_size}_nc{n_conv}_hf{h_fea_len}/"
    os.makedirs(trial_output_dir, exist_ok=True)

    log_file = os.path.join(trial_output_dir, "training_log.txt")

    # Construct the main.py command
    command = [
        "python", "main.py",
        data_dir_for_main_py,
        "--task", "regression",
        "--epochs", str(epochs_per_trial), # Use reduced epochs
        "--lr", str(lr),
        "-b", str(batch_size),
        "--n-conv", str(n_conv),
        "--h-fea-len", str(h_fea_len),
        "-j", str(num_data_workers), # Add data loading workers argument
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2"
    ]

    with open(log_file, "w") as outfile:
        try:
            print(f"Executing: {' '.join(command)}")
            subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT)
            print(f"Trial {i+1} completed. Results in {trial_output_dir}")

            # Move model files to the trial-specific directory
            if os.path.exists("model_best.pth.tar"):
                shutil.move("model_best.pth.tar", os.path.join(trial_output_dir, "model_best.pth.tar"))
            if os.path.exists("checkpoint.pth.tar"):
                shutil.move("checkpoint.pth.tar", os.path.join(trial_output_dir, "checkpoint.pth.tar"))
            if os.path.exists("test_results.csv"):
                shutil.move("test_results.csv", os.path.join(trial_output_dir, "test_results.csv"))

        except subprocess.CalledProcessError as e:
            print(f"Trial {i+1} failed with error code {e.returncode}. Command: {' '.join(command)}")
            print(f"Check {log_file} for details.")
        except FileNotFoundError:
            print(f"Error: 'python' or 'main.py' command not found. Check your PATH and current directory.")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Command attempted: {' '.join(command)}")

print("\n--- Hyperparameter Tuning Process Finished ---")
