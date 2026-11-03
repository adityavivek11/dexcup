"""Train the teacher on a Modal cloud GPU (used when the free Colab GPU allowance ran out, 3 Nov).

Same train_teacher.py as everywhere else; Modal runs it in a container with a T4 and returns the model.
WHY these settings:
  - torch pinned to the laptop's version (2.14.1): the model is saved as a whole module, and loading it
    back on the laptop for the sim evaluation is safest with the identical version.
  - hard timeout (45 min, ~$0.45 on a T4 at ~$0.59/h): a hung job can't drain the small credit.
  - the model is the function's RETURN value: nothing stays running on Modal afterwards, so there is no
    session to forget to stop (the lost Colab cross-validation run, 3 Nov, was exactly that failure).

Run:  python -m modal run modal_train.py                 (full teacher, 80 epochs -> models/teacher.pt)
      python -m modal run modal_train.py --fold 2 --epochs 40   (cross-validation fold -> teacher_fold2.pt)
"""
from pathlib import Path
import modal

ROOT = Path(__file__).parent
REMOTE = "/root/dexcup"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "numpy")
         .add_local_file(ROOT / "policy.py", f"{REMOTE}/policy.py")
         .add_local_file(ROOT / "train_teacher.py", f"{REMOTE}/train_teacher.py")
         .add_local_file(ROOT / "data" / "split.json", f"{REMOTE}/data/split.json")
         .add_local_dir(ROOT / "data" / "replayed", f"{REMOTE}/data/replayed")
         .add_local_dir(ROOT / "data" / "dart", f"{REMOTE}/data/dart"))
app = modal.App("dexcup-teacher", image=image)


@app.function(gpu="T4", timeout=45 * 60)
def train(args: list, name: str) -> bytes:
    import os, subprocess, sys
    os.chdir(REMOTE)
    subprocess.run([sys.executable, "-u", "train_teacher.py", *args], check=True)
    return open(f"models/{name}", "rb").read()


@app.local_entrypoint()
def main(epochs: int = 80, fold: int = -1):
    args = ["--dart", "--epochs", str(epochs)] + (["--fold", str(fold)] if fold >= 0 else [])
    name = "teacher.pt" if fold < 0 else f"teacher_fold{fold}.pt"
    data = train.remote(args, name)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / name).write_bytes(data)
    print(f"saved models/{name} ({len(data)/1e6:.1f} MB)")
