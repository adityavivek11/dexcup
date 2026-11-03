"""Write one Kaggle notebook folder per training job (kaggle/k_<name>/), all reading the same private dataset.

WHY one notebook per job: Kaggle runs notebooks independently, so the velocity experiments and the
cross-validation of v3 train at the same time instead of one after another. Each job's command lines are
baked into its script; outputs (models/*.pt, train.log) are fetched with `kaggle kernels output`.
Kaggle keeps outputs on its servers after the run, so a sleeping laptop can't lose them (the Colab
cross-validation run was lost that way, 3 Nov).

Run: python kaggle/make_kernels.py   then, from INSIDE each folder: kaggle kernels push -p . --accelerator NvidiaTeslaT4
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
USER = "adityavivek11"
JOBS = {
    "pv": ["--dart --epochs 80 --frame palm --vel --name teacher_pv.pt"],
    "ov": ["--dart --epochs 80 --frame object --vel --name teacher_ov.pt"],
    "cv-v3": [f"--dart --epochs 40 --frame palm --fold {k}" for k in range(5)],
}

SCRIPT = '''# Runs ON Kaggle (GPU): unpack the dexcup bundle, run each train_teacher.py command, keep the models.
import glob, os, shutil, subprocess, sys, zipfile
JOBS = {jobs!r}
src = glob.glob("/kaggle/input/**/dexcup_cloud.zip", recursive=True)
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
        log.write("=== " + args + "\\n"); log.flush()
        r = subprocess.run([sys.executable, "-u", "train_teacher.py", *args.split()], stdout=log, stderr=subprocess.STDOUT)
        rc = rc or r.returncode
        for f in glob.glob("models/*.pt"):          # copy after EVERY job: a later failure can't lose earlier models
            shutil.copy(f, "/kaggle/working/")
print(open("/kaggle/working/train.log").read()[-4000:])
sys.exit(rc)
'''

if __name__ == "__main__":
    for name, jobs in JOBS.items():
        d = HERE / f"k_{name}"
        d.mkdir(exist_ok=True)
        (d / "train_kernel.py").write_text(SCRIPT.format(jobs=jobs))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/dexcup-{name}", "title": f"dexcup-{name}", "code_file": "train_kernel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": False, "dataset_sources": [f"{USER}/dexcup-replayed"]}, indent=1))
        print("wrote", d)
