import random
import time
import speech_recognition as sr
import handy1

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    recognizer.dynamic_energy_threshold =True 
    with microphone as source:
        recognizer.energy_threshold=1500
        recognizer.pause_threshold=0.5
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }


    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response


def speech_main():
    NUM_GUESSES = 300
    PROMPT_LIMIT = 20

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    for i in range(NUM_GUESSES):
        for j in range(PROMPT_LIMIT):
            print('{}. Speak!'.format(i+1))
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. What did you say?\n")

        print("You said: {}".format(guess["transcription"]))
        if guess["transcription"]==None:
            continue
        guess_is_correct = guess["transcription"].lower() 
        user_has_more_attempts = i < NUM_GUESSES - 1

        print(guess_is_correct)

        if guess_is_correct.find('zoom in') != -1 or  guess_is_correct.find('zoomin') != -1:
            print("zoom in -100")
            handy1.action_func('zoomin')
        elif guess_is_correct.find('zoom out') != -1 or  guess_is_correct.find('zoomout') != -1:
            print ('zoom out +100')
            handy1.action_func('zoomout')
        elif guess_is_correct.find('terminal') != -1 or  guess_is_correct.find('cmd') != -1:
            print ('open terminal')
            handy1.action_func('terminal')
        elif guess_is_correct.find('close tab')!=-1 or guess_is_correct.find('closetab')!=-1:
            print('close tab')
            handy1.action_func('closetab')
        elif guess_is_correct.find('open tab')!=-1 or guess_is_correct.find('opentab')!=-1 or guess_is_correct.find('newtab')!=-1 or guess_is_correct.find('new tab')!=-1:
            print('open tab')
            handy1.action_func('opentab')
        elif guess_is_correct.find('close window')!=-1 or guess_is_correct.find('closewindow')!=-1:
            print('close window')
            handy1.action_func('closewindow')
        elif guess_is_correct.find('screenshot')!=-1 or guess_is_correct.find('screen shot')!=-1:
            print('screen shot')
            handy1.action_func('screenshot')
        elif guess_is_correct.find('switch tab')!=-1 or guess_is_correct.find('switchtab')!=-1:
            print('switch tab')
            handy1.action_func('switchtab')
        elif guess_is_correct.find('switch window')!=-1 or guess_is_correct.find('switchwindow')!=-1:
            print('switch window')
            handy1.action_func('switchwindow')
        
speech_main()
