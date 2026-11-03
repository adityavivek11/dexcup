# Runs ON the Colab VM: is training still running, and what has it printed so far?
import os
pid = int(open("/content/train.pid").read())
alive = os.path.exists(f"/proc/{pid}") and "zombie" not in open(f"/proc/{pid}/status").read().lower()
lines = open("/content/train.log").read().splitlines()
print("RUNNING" if alive else "FINISHED", "| model saved:", os.path.exists("/content/dexcup/models/teacher.pt"))
print("\n".join(l for l in lines if "Warning" not in l and "self.encoder" not in l)[-1800:])
