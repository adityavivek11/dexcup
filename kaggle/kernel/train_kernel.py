# Runs ON Kaggle (GPU notebook): unpack the dexcup bundle and train the teacher. Pushed by
# `kaggle kernels push`; outputs (teacher.pt, train.log) are fetched with `kaggle kernels output`.
# Kaggle may hand over the dataset's zip extracted or as-is, so both cases are handled.
import glob, os, shutil, subprocess, sys, zipfile
src = glob.glob("/kaggle/input/**/dexcup_cloud.zip", recursive=True)
if src:
    zipfile.ZipFile(src[0]).extractall("/kaggle/working")
else:
    d = glob.glob("/kaggle/input/**/dexcup/policy.py", recursive=True)[0]
    shutil.copytree(os.path.dirname(d), "/kaggle/working/dexcup")
os.chdir("/kaggle/working/dexcup")
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "| torch", torch.__version__, flush=True)
args = os.environ.get("DEXCUP_ARGS", "--dart --epochs 80").split()
with open("/kaggle/working/train.log", "w") as log:
    r = subprocess.run([sys.executable, "-u", "train_teacher.py", *args], stdout=log, stderr=subprocess.STDOUT)
print(open("/kaggle/working/train.log").read()[-3000:])
for f in glob.glob("models/*.pt"):
    shutil.copy(f, "/kaggle/working/")
sys.exit(r.returncode)
