import os, subprocess

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

arguments = "main.py --train --src_lang en --target ja --gpu {gpu} --layers_p {layer}  --proj_prob 0.0 --policy cosine_src_inv --seed {seed} --epochs 20 --sig_center 3750"
arguments2 = "main.py --train --src_lang en --target {lang} --gpu {gpu} --layers_p 6  --proj_prob 0.5 --policy random --seed {seed} --epochs 10 --sig_center 3750"
arguments3 = "main.py --train --src_lang en --target {lang} --gpu {gpu} --layers_p {layer}  --proj_prob 0.0 --policy cosine_src_inv --seed {seed} --epochs 10 --sig_center 3750"
arguments4 = "main.py --train --src_lang en --target {lang} --gpu {gpu} --layers_p {layer}  --proj_prob {prob} --policy random --seed {seed} --epochs 7 --sig_center 3750"


pid_to_gpu = {}
gpu_to_pid = {}
free_gpus = [0, 1, 2, 3]


layers = [i for i in range(12)]
seeds = [123, 456, 789]

flag = False

count = 0

langs = ['hi', 'id']
probs = [0.7, 0.9, 1.0]

for prob in probs:
    for lang in langs:
        for layer in layers:
            for seed in seeds:

                found_gpus = [] 

                print("Waiting for a gpu to be freed......")

                while len(free_gpus) == 0:

                    for pid in pid_to_gpu.keys():
                        
                        if pid.poll() is not None:
                            found_gpus.append(pid_to_gpu[pid])

                    for gpu_t in found_gpus:
                        pid_to_gpu.pop(gpu_to_pid[gpu_t])
                        gpu_to_pid.pop(gpu_t)
                        free_gpus.append(gpu_t)

                gpu = free_gpus[0]
                free_gpus.remove(gpu)
                args = arguments4.format(gpu=str(gpu), layer=str(layer), seed=str(seed), lang = lang, prob=str(prob))
                all_args = args.split()

                proc = subprocess.Popen(["nohup", "python3", all_args[0], all_args[1], all_args[2],  all_args[3], all_args[4], all_args[5], all_args[6], all_args[7], all_args[8], all_args[9], all_args[10], all_args[11], all_args[14], all_args[15], all_args[16], all_args[17], all_args[18], all_args[19], all_args[12], all_args[13]])
                # proc = subprocess.Popen(["nohup", "python3", all_args[0], " ".join(all_args[1:])])
                pid_t = proc.pid # <--- access `pid` attribute to get the pid of the child process.

                pid_to_gpu[proc] = gpu
                gpu_to_pid[gpu] = proc

                print(gpu_to_pid)

print("Exiting")