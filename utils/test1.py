def main(): 
    print("Hello World!")

if __name__ == "__main__": 
    main() 


'''

Sub RunPythonScript()
    Dim pythonExe As String
    Dim scriptPath As String
    Dim shellOutput As String
    Dim cmd As String
    Dim wsh As Object
    Dim process As Object
    Dim stdout As Object
    
    ' Specify the path to the Python executable
    pythonExe = "C:\Program Files\Python312\python.exe"
    
    ' Specify the path to the Python script
    scriptPath = "C:\Users\chena\Desktop\code\backtesterv2\utils\test1.py"
    
    ' Construct the command to run the Python script
    cmd = pythonExe & " """ & scriptPath & """"
    
    ' Use Windows Script Host to execute the command and capture the output
    On Error GoTo ErrorHandler
    Set wsh = CreateObject("WScript.Shell")
    Set process = wsh.Exec(cmd)
    Set stdout = process.stdout
    
    ' Capture the Python script's output
    shellOutput = stdout.ReadAll
    
    ' Display the result in a message box
    MsgBox shellOutput, vbInformation, "Python Script Output"
    
    Exit Sub

ErrorHandler:
    MsgBox "An error occurred while running the Python script: " & Err.Description, vbCritical, "Error"
End Sub



'''