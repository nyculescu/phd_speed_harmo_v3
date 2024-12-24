import os
import subprocess

def generate_ssh_for_github():
    ssh_dir = os.path.expanduser("~/.ssh")
    os.makedirs(ssh_dir, exist_ok=True)

    key_path = os.path.join(ssh_dir, "id_rsa")
    subprocess.run(["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", key_path, "-N", ""], check=True)

    public_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if os.path.exists(public_key_path):
        with open(public_key_path, "r") as pub_key_file:
            public_key = pub_key_file.read()
        print(public_key)
    else:
        print("Public key not found.")

if __name__ == '__main__':
    generate_ssh_for_github()