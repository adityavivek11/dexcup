# Runs ON the Colab VM: progress of the cross-validation run.
import os
pid = int(open("/content/cv.pid").read())
alive = os.path.exists(f"/proc/{pid}") and "zombie" not in open(f"/proc/{pid}/status").read().lower()
lines = [l for l in open("/content/cv.log").read().splitlines() if l.startswith(("episodes", "best", "Traceback", "epoch"))]
done = [f for f in range(5) if os.path.exists(f"/content/dexcup/models/teacher_fold{f}.pt")]
print("RUNNING" if alive else "FINISHED", "| folds with a saved model:", done)
print("\n".join(lines[-3:]))
