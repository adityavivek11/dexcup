# Runs ON the Colab VM (via `colab exec -f cloud/start_train.py`): unpack the bundle and start teacher
# training in the BACKGROUND. WHY background: `colab exec` calls time out (default 30 s) and training
# takes minutes; a detached process keeps running, and cloud/poll_train.py reads its log.
import os, subprocess, zipfile, torch
zipfile.ZipFile("/content/dexcup_cloud.zip").extractall("/content")
os.chdir("/content/dexcup")
print("episodes:", len(os.listdir("data/replayed")), "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
log = open("/content/train.log", "w")
p = subprocess.Popen(["python", "-u", "train_teacher.py", "--epochs", "80", "--dart"], stdout=log, stderr=subprocess.STDOUT,
                     cwd="/content/dexcup", start_new_session=True)
open("/content/train.pid", "w").write(str(p.pid))
print("training started, pid", p.pid)
