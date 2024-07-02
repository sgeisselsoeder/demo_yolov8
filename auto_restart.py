from subprocess import Popen
from time import sleep
import pyautogui


while True:
    print('Starting subprocess')
    p = Popen(['python3', 'main.py']) # something long running
    sleep(12)
    print('Maxmimizing window')
    # pyautogui.hotkey('winleft', 'up')
    # pyautogui.hotkey('alt', 'tab')
    # pyautogui.hotkey('winleft', 'up')
    pyautogui.keyDown('winleft')
    pyautogui.press('up')
    pyautogui.keyUp('winleft')
    time_to_run = 17
    # time_to_run = 10*60
    print("Sleeping for ", time_to_run, " seconds")
    sleep(time_to_run)
    # ... do other stuff while subprocess is running
    print('Terminating subprocess now')
    p.terminate()
    print('Subprocess terminated')
    sleep(2)