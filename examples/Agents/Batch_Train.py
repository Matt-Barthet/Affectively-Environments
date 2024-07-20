import subprocess
import os

directory = os.path.dirname(os.path.realpath(__file__))

# for cluster in [0, 1, 2, 3]:
for run in range(1, 3):
    for weight in [0, 0.5, 1]:
        # for target_name, target_signal in [("Minimize", "np.zeros"), ("Maximize", "np.ones"), ("imitate", "imitate")]:
        script_path = '.Examples/Agents/Train_PPO.py'
        command = f'cd {directory} && conda activate unity_gym && python {script_path} {run} {weight}'
        subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
