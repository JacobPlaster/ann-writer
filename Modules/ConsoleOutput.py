HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

_IsLogging = True

class ConsoleOutput:
    # prints to the console in a color

    def printYellow(inOutput):
        if(_IsLogging):
            print(WARNING + inOutput + ENDC)
    def printGreen(inOutput):
        if(_IsLogging):
            print(OKGREEN + inOutput + ENDC)
    def printUnderline(inOutput):
        if(_IsLogging):
            print(UNDERLINE + inOutput + ENDC)
    def printRed(inOutput):
        if(_IsLogging):
            print(FAIL + inOutput + ENDC)
    def printBold(inOutput):
        if(_IsLogging):
            print(BOLD + inOutput + ENDC)
