# Runs ON Kaggle (GPU): unpack the dexcup bundle, run each train_teacher.py command, keep the models.
import glob, os, shutil, subprocess, sys, zipfile
JOBS = ['--dart --epochs 80 --frame palm --vel --name teacher_pv.pt']
src = glob.glob("/kaggle/input/**/dexcup_cloud*.zip", recursive=True)
if src:
    zipfile.ZipFile(src[0]).extractall("/kaggle/working")
else:
    d = glob.glob("/kaggle/input/**/dexcup/policy.py", recursive=True)[0]
    shutil.copytree(os.path.dirname(d), "/kaggle/working/dexcup")
os.chdir("/kaggle/working/dexcup")
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
rc = 0
with open("/kaggle/working/train.log", "w") as log:
    for args in JOBS:
        log.write("=== " + args + "\n"); log.flush()
        r = subprocess.run([sys.executable, "-u", "train_teacher.py", *args.split()], stdout=log, stderr=subprocess.STDOUT)
        rc = rc or r.returncode
        for f in glob.glob("models/**/*.pt", recursive=True):          # copy after EVERY job: a later failure can't lose earlier models
            shutil.copy(f, "/kaggle/working/")
print(open("/kaggle/working/train.log").read()[-4000:])
sys.exit(rc)
