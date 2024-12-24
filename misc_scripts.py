import os
import subprocess

def generate_ssh_for_github():
    ssh_dir = os.path.expanduser("~/.ssh")
    os.makedirs(ssh_dir, exist_ok=True)

    key_path = os.path.join(ssh_dir, "id_rsa")
    subprocess.run(["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", key_path, "-N", ""], check=True)

def authorize_ssh_for_github():
    # Start the SSH agent and capture its output
    ssh_agent_output = subprocess.check_output(['ssh-agent', '-s'], text=True)

    # Parse and set the SSH_AUTH_SOCK and SSH_AGENT_PID environment variables
    for line in ssh_agent_output.splitlines():
        if 'SSH_AUTH_SOCK' in line or 'SSH_AGENT_PID' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value.rstrip(';')

    print("SSH agent started and environment variables set.")
    ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
    result = subprocess.run(['ssh-add', ssh_key_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("SSH private key added successfully.")
    else:
        print(f"Error adding SSH private key: {result.stderr}")

    ##########################
    # subprocess.run(['eval', '$(ssh-agent -s)'], shell=True, check=True)
    # print("SSH agent started.")

    # ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
    # result = subprocess.run(['ssh-add', ssh_key_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # if result.returncode == 0:
    #     print("SSH private key added successfully.")
    # else:
    #     print(f"Error adding SSH private key: {result.stderr}")
    
    # ssh_dir = os.path.expanduser("~/.ssh")
    # os.chmod(ssh_dir, 0o700)
    # os.chmod(os.path.join(ssh_dir, "id_rsa"), 0o600)

    ##########################
    # subprocess.run(["git", "config", "--global", "url.ssh://git@github.com/.insteadOf", "https://github.com/"])
    # subprocess.run(["eval", "$(ssh-agent -s)"], shell=True)
    # subprocess.run(["ssh-add", os.path.expanduser("~/.ssh/id_rsa")])
    # with open(os.path.expanduser("~/.ssh/id_rsa.pub"), "r") as pub_key_file:
    #     public_key = pub_key_file.read()
    # print(public_key)

if __name__ == '__main__':
    # generate_ssh_for_github()
    authorize_ssh_for_github()