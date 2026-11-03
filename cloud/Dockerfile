# Linux container for Google's Colab CLI (github.com/googlecolab/google-colab-cli), which supports only
# Linux/macOS; the laptop is Windows. Used only to drive cloud GPU training, never for the sim.
#
# Build:  docker build -t dexcup-colab cloud/
# Run:    docker run --rm -v colab-home:/root -v "<project>:/work" dexcup-colab colab <command>
#   colab-home : named volume = the container's home, so the sign-in token and the CLI's session list
#                survive between `docker run`s (sign in once, not on every command)
#   /work      : the project folder, so uploads/downloads go straight to/from the repo
FROM python:3.12-slim
# keyrings.alt: containers have no OS keyring; without a backend the CLI would re-run the sign-in flow on
# every command. PlaintextKeyring stores the token as a file in the colab-home volume (local Docker only).
RUN pip install --no-cache-dir google-colab-cli keyrings.alt
ENV PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
WORKDIR /work
