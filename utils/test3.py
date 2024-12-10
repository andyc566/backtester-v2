import sys

def printcellvalue(inputcell):
    outputcell = inputcell + 1
    return outputcell

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputcell = int(sys.argv[1])  # Convert the argument to an integer
        result = printcellvalue(inputcell)
        print(result)  # This will be the output printed to stdout
    else:
        print("No argument passed. Please provide an input value.")
