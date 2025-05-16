import paramiko
import subprocess
from multiprocessing import Pool


def work_to_execute_at_distance(pc_name, host, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(host, username=username, password=password)

        #---------------------------------------------------

        cmd1 = "cd ~/Documents/flagAutomata/src && ./launch.sh" # launch the flagAutomata code
        cmd2 = f'sshpass -p {password} rsync -avz {username}@{host}:/home/loi/Documents/flagAutomata/data_plots/simulationAnalysis/ /home/kalessia/flagAutomata/data_plots/simulationAnalysis/'
        cmd3 = "pkill -f python3" # kill python processes

        cmd_to_execute = [cmd2]

        #---------------------------------------------------

        for cmd in cmd_to_execute:

            if cmd.startswith("sshpass"):
                result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                display_std(pc_name, result.stdout, result.stderr)

            else:
                _, stdout, stderr = client.exec_command(cmd)
                display_std(pc_name, stdout.read(), stderr.read())
        
    #---------------------------------------------------

    except Exception as e:
        print(f"Exception raised on {pc_name}: {e}")

    #---------------------------------------------------
    
    finally:
        client.close()
        print(f"Connexion closed on {pc_name}")


#---------------------------------------------------

def display_std(pc_name, stdout, stderr):
    stdout_content = stdout.decode('utf-8')
    stderr_content = stderr.decode('utf-8')

    print(f"stdout_content on {pc_name}:\n{stdout_content}")
    if stderr_content:
        print(f"stderr_content on {pc_name}:\n{stderr_content}")


###########################################################################
# Parallelization
###########################################################################

def parallelize_processes(pc_list):

    with Pool() as pool:
        pool.map(worker, pc_list)
        pool.close() # no more tasks will be submitted to the pool
        pool.join() # wait for all processes to complete

#---------------------------------------------------

def worker(task):
    pc_name, host, username, password = task
    work_to_execute_at_distance(pc_name, host, username, password)


###########################################################################
# MAIN

# Launch with "python3 script_ssh.py &" from "flagAutomata/src"
###########################################################################

if __name__ == "__main__":
    
    # PC list: (pc_name, host, username, password)
    pc_list = [
        ("Dell PC", "192.168.1.18", "loi", "dellSchwarz14!")
        # ("Dell PC", "192.168.1.124", "loi", "dellSchwarz14!")
        # ("192.168.1.101", "utente2", "password2")  # multi-arena
    ]

    parallelize_processes(pc_list)
