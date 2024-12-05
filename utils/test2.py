import os 

def main(): 
    filename = os.path.normpath("C:/Users/chena/Desktop/code/iso-ne-energy-forecast/data/plots/predictions_plot.png")
    print(filename)

if __name__ == "__main__": 
    main() 

'''
Sub TestImageInsert()
    Dim imagePath As String
    Dim ws As Worksheet
    Dim imgLeft As Double
    Dim imgTop As Double
    
    imagePath = "C:\Users\chena\Desktop\code\iso-ne-energy-forecast\data\plots\predictions_plot.png"
    
    ' Confirm the file exists
    If Dir(imagePath) = "" Then
        MsgBox "File not found: " & imagePath, vbCritical, "Error"
        Exit Sub
    End If
    
    ' Insert the image
    Set ws = ThisWorkbook.ActiveSheet
    imgLeft = ws.Range("A1").Left
    imgTop = ws.Range("A1").Top
    
    On Error Resume Next
    ws.Pictures.Insert(imagePath).Select
    If Err.Number <> 0 Then
        MsgBox "Error inserting image: " & Err.Description, vbCritical, "Error"
    Else
        Selection.ShapeRange.LockAspectRatio = msoTrue ' Maintain aspect ratio
        Selection.ShapeRange.Left = imgLeft
        Selection.ShapeRange.Top = imgTop
        MsgBox "Image inserted successfully!", vbInformation, "Success"
    End If
    On Error GoTo 0
End Sub



'''