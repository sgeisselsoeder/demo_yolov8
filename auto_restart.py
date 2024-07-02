from subprocess import Popen
from time import sleep

while True:
    print('Starting subprocess')
    p = Popen(['python3', 'main.py']) # something long running
    time_to_run = 15
    time_to_run = 10*60
    print("Sleeping for ", time_to_run, " seconds")
    sleep(time_to_run)
    # ... do other stuff while subprocess is running
    print('Terminating subprocess now')
    p.terminate()
    print('Subprocess terminated')
    sleep(2)