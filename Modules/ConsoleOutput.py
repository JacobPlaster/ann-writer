HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class ConsoleOutput:
    # prints to the console in a color

    def printYellow(inOutput):
        print(WARNING + inOutput + ENDC)
    def printGreen(inOutput):
        print(OKGREEN + inOutput + ENDC)
    def printUnderline(inOutput):
        print(UNDERLINE + inOutput + ENDC)
    def printRed(inOutput):
        print(FAIL + inOutput + ENDC)
    def printBold(inOutput):
        print(BOLD + inOutput + ENDC)
