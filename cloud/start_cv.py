# Runs ON the Colab VM: 5-fold cross-validation, the folds trained one after another in ONE background
# process (40 epochs each: the loss plateaus by ~40). Logs to /content/cv.log; poll with cloud/poll_cv.py.
import os, subprocess, zipfile
zipfile.ZipFile("/content/dexcup_cloud.zip").extractall("/content")
os.chdir("/content/dexcup")
cmd = " && ".join(f"python -u train_teacher.py --dart --epochs 40 --fold {k}" for k in range(5))
log = open("/content/cv.log", "w")
p = subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open("/content/cv.pid", "w").write(str(p.pid))
print("cross-validation started, pid", p.pid)
