import psutil, time
import subprocess
import argparse

def get_memory_usage(log_file=None, stdout=False):
    memory_usage = psutil.virtual_memory().percent
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"memory usage: {memory_usage}%\n")

    if stdout:
        print(f"memory usage: {memory_usage}%")

def get_gpu_utilization(log_file=None, stdout=False):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        for idx, line in enumerate(lines):
            gpu_util, mem_used, mem_total = map(str.strip, line.split(','))

            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(f"🖥  GPU {idx}: {gpu_util}% load | {mem_used} MiB / {mem_total} MiB\n")
            if stdout:
                print(f"🖥 GPU {idx}: {gpu_util}% load | {mem_used} MiB / {mem_total} MiB")
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Make sure NVIDIA drivers are installed.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='gigaam_checkpoints/resource_usage.log', help='set a path to logfile')
    parser.add_argument('--stdout', action='store_true', help='indicate if we want to print log in stdoutput')
    args = parser.parse_args()

    logfile_path = args.logfile

    try:
        while True:
            get_memory_usage(logfile_path, args.stdout)
            get_gpu_utilization(logfile_path, args.stdout)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
