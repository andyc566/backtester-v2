'''
Sub RunPythonScript()
    Dim pythonExe As String
    Dim pythonScript As String
    Dim argument As String
    Dim tempFile As String
    Dim result As String
    Dim shellCommand As String
    Dim objFSO As Object
    Dim objFile As Object
    
    ' Define the path to the Python executable
    pythonExe = "C:\Path\To\python.exe" ' Adjust this to your Python executable path

    ' Define the path to the Python script
    pythonScript = "C:\Path\To\your_script.py" ' Adjust this to your Python script path

    ' Get the argument from cell A1
    argument = Range("A1").Value

    ' Define a temporary file to store the Python output
    tempFile = Environ("TEMP") & "\python_output.txt"
    
    ' Build the shell command to execute the Python script with an argument
    shellCommand = pythonExe & " " & pythonScript & " " & argument & " > " & tempFile

    ' Run the shell command
    Shell shellCommand, vbHide

    ' Wait for the Python script to finish
    Application.Wait (Now + TimeValue("0:00:02"))

    ' Read the output from the temporary file
    Set objFSO = CreateObject("Scripting.FileSystemObject")
    If objFSO.FileExists(tempFile) Then
        Set objFile = objFSO.OpenTextFile(tempFile, 1)
        result = objFile.ReadAll
        objFile.Close
    Else
        MsgBox "Output file not found.", vbExclamation
        Exit Sub
    End If

    ' Place the result in cell A2
    Range("A2").Value = result

    ' Clean up
    objFSO.DeleteFile tempFile
End Sub

'''