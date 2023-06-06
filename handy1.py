import pyautogui
import time
import winsound
frequency = 2500  
duration = 1000  

def action_func(incoming_data):

    print('handy', incoming_data)

    if 'zoomin' in incoming_data:
        pyautogui.hotkey('ctrl', '+')
        # time.sleep(1)

    if 'zoomout' in incoming_data:
        pyautogui.hotkey('ctrl', '-')
        # time.sleep(1)

    if 'terminal' in incoming_data:
        pyautogui.hotkey('winleft','r')
        pyautogui.press('enter')

    if 'closetab' in incoming_data:
        pyautogui.hotkey('ctrl','w')
        # pyautogui.press('enter')
    
    if 'opentab' in incoming_data:
        pyautogui.hotkey('ctrl','t')
        # pyautogui.press('enter')
    
    if 'closewindow' in incoming_data:
        pyautogui.hotkey('alt','f4')

    if 'screenshot' in incoming_data:
        # pyautogui.hotkey('winleft','')
        pyautogui.screenshot(r"C:\Users\HP\OneDrive\Desktop\Capstone\Screenshots\screenshot.png")
        winsound.Beep(frequency, duration)

    if 'switchtab' in incoming_data:
        pyautogui.hotkey('ctrl','tab')
    
    if 'switchwindow' in incoming_data:
        pyautogui.hotkey('alt','tab')

    
    incoming_data = ""                            # clears the data
