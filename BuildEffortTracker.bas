Attribute VB_Name = "BuildEffortTracker"
Option Explicit

' ============================================================
' Project Effort Tracker builder
' Run CreateEffortTracker() on a blank workbook.
' Alt+F11 -> Insert -> Module -> paste this -> F5 (or run CreateEffortTracker)
' ============================================================

Const START_DATE As Date = #7/3/2026#
Const END_DATE As Date = #8/4/2026#
Const MAX_RES As Long = 50
Const MAX_TIX As Long = 50
Const LAST_LOG_ROW As Long = 1001

Sub CreateEffortTracker()

    Dim wb As Workbook
    Dim wsInstr As Worksheet, wsRes As Worksheet, wsTix As Worksheet
    Dim wsLog As Worksheet, wsMat As Worksheet, wsSum As Worksheet
    Dim ws As Worksheet

    Application.ScreenUpdating = False
    Application.DisplayAlerts = False

    Set wb = ActiveWorkbook

    ' remove all existing sheets except one, we will rename/reuse it
    Do While wb.Worksheets.Count > 1
        wb.Worksheets(wb.Worksheets.Count).Delete
    Loop
    wb.Worksheets(1).Cells.Clear
    wb.Worksheets(1).Name = "Instructions"
    Set wsInstr = wb.Worksheets(1)

    Set wsRes = wb.Worksheets.Add(After:=wsInstr): wsRes.Name = "Resources"
    Set wsTix = wb.Worksheets.Add(After:=wsRes): wsTix.Name = "Jira Tickets"
    Set wsLog = wb.Worksheets.Add(After:=wsTix): wsLog.Name = "Effort Log"
    Set wsMat = wb.Worksheets.Add(After:=wsLog): wsMat.Name = "Daily Tracker Matrix"
    Set wsSum = wb.Worksheets.Add(After:=wsMat): wsSum.Name = "Ticket Summary"

    BuildInstructions wsInstr
    BuildResources wsRes
    BuildTickets wsTix
    BuildEffortLog wsLog
    BuildMatrix wsMat
    BuildTicketSummary wsSum

    wsInstr.Activate
    Application.DisplayAlerts = True
    Application.ScreenUpdating = True

    MsgBox "Effort tracker built. Fill Resources and Jira Tickets first, then use Effort Log daily.", vbInformation

End Sub

' ------------------------------------------------------------
Private Sub StyleHeader(ws As Worksheet, r As Long, colFrom As Long, colTo As Long)
    With ws.Range(ws.Cells(r, colFrom), ws.Cells(r, colTo))
        .Font.Name = "Arial"
        .Font.Bold = True
        .Font.Color = RGB(255, 255, 255)
        .Interior.Color = RGB(31, 78, 120)
        .HorizontalAlignment = xlCenter
        .VerticalAlignment = xlCenter
        .WrapText = True
        .Borders.LineStyle = xlContinuous
        .Borders.Color = RGB(191, 191, 191)
    End With
End Sub

Private Sub ShadeInput(rng As Range)
    rng.Interior.Color = RGB(255, 255, 204)
    rng.Borders.LineStyle = xlContinuous
    rng.Borders.Color = RGB(191, 191, 191)
    rng.Font.Name = "Arial"
    rng.Font.Size = 10
End Sub

' ------------------------------------------------------------
Private Sub BuildInstructions(ws As Worksheet)
    ws.Columns("A").ColumnWidth = 110
    ws.Range("A1").Value = "Project Effort Tracker - How to use this workbook"
    ws.Range("A1").Font.Bold = True
    ws.Range("A1").Font.Size = 14
    ws.Range("A1").Font.Name = "Arial"

    Dim lines As Variant
    lines = Array( _
        "", _
        "This workbook has 5 tabs:", _
        "", _
        "1. Resources - Master list of all developers and QA (onshore + offshore). Fill this in first.", _
        "2. Jira Tickets - Master list of all ~50 Jira tickets / scope items. Fill this in second.", _
        "3. Effort Log - The DAILY ENTRY sheet. One row per resource per ticket per day worked. Location and Role auto-fill from Resources once you pick a Resource Name. Multiple people can log the same ticket on the same day.", _
        "4. Daily Tracker Matrix - Auto-calculated. Total hours per resource per day (03-Jul to 04-Aug). RED = resource logged zero hours that day, chase it up. GREEN = hours logged.", _
        "5. Ticket Summary - Auto-calculated. Total effort and number of distinct contributors per Jira ticket.", _
        "", _
        "Only edit YELLOW shaded cells. Everything else is a formula." _
    )

    Dim i As Long
    For i = LBound(lines) To UBound(lines)
        ws.Cells(i + 2, 1).Value = lines(i)
        ws.Cells(i + 2, 1).Font.Name = "Arial"
        ws.Cells(i + 2, 1).Font.Size = 10
    Next i
End Sub

' ------------------------------------------------------------
Private Sub BuildResources(ws As Worksheet)
    ws.Range("A1").Value = "Resource Name"
    ws.Range("B1").Value = "Location (Onshore/Offshore)"
    ws.Range("C1").Value = "Role (Dev/QA)"
    StyleHeader ws, 1, 1, 3

    Dim sample As Variant
    sample = Array( _
        Array("Amit Sharma", "Offshore", "Dev"), _
        Array("Priya Nair", "Offshore", "QA"), _
        Array("John Miller", "Onshore", "Dev"), _
        Array("Sara Lopez", "Onshore", "QA") _
    )
    Dim i As Long, r As Long
    r = 2
    For i = LBound(sample) To UBound(sample)
        ws.Cells(r, 1).Value = sample(i)(0)
        ws.Cells(r, 2).Value = sample(i)(1)
        ws.Cells(r, 3).Value = sample(i)(2)
        r = r + 1
    Next i

    ShadeInput ws.Range("A2:C" & (MAX_RES + 1))

    With ws.Range("B2:B" & (MAX_RES + 1)).Validation
        .Delete
        .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, Formula1:="Onshore,Offshore"
    End With
    With ws.Range("C2:C" & (MAX_RES + 1)).Validation
        .Delete
        .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, Formula1:="Dev,QA"
    End With

    ws.Columns("A").ColumnWidth = 26
    ws.Columns("B").ColumnWidth = 26
    ws.Columns("C").ColumnWidth = 18
    ws.Rows(1).RowHeight = 30
    ws.Application.ActiveWindow.FreezePanes = False
    ws.Range("A2").Select
    ws.Application.ActiveWindow.FreezePanes = True
End Sub

' ------------------------------------------------------------
Private Sub BuildTickets(ws As Worksheet)
    ws.Range("A1").Value = "Jira Ticket"
    ws.Range("B1").Value = "Scope Item / Task Description"
    StyleHeader ws, 1, 1, 2

    Dim sample As Variant
    sample = Array( _
        Array("PROJ-101", "Build room mapping ingestion service"), _
        Array("PROJ-102", "QA regression for booking flow"), _
        Array("PROJ-103", "API contract review") _
    )
    Dim i As Long, r As Long
    r = 2
    For i = LBound(sample) To UBound(sample)
        ws.Cells(r, 1).Value = sample(i)(0)
        ws.Cells(r, 2).Value = sample(i)(1)
        r = r + 1
    Next i

    ShadeInput ws.Range("A2:B" & (MAX_TIX + 1))

    ws.Columns("A").ColumnWidth = 18
    ws.Columns("B").ColumnWidth = 55
    ws.Application.ActiveWindow.FreezePanes = False
    ws.Range("A2").Select
    ws.Application.ActiveWindow.FreezePanes = True
End Sub

' ------------------------------------------------------------
Private Sub BuildEffortLog(ws As Worksheet)
    Dim headers As Variant
    headers = Array("Date", "Resource Name", "Location", "Role", "Jira Ticket", "Effort (Hours)", "Comments")
    Dim c As Long
    For c = 1 To 7
        ws.Cells(1, c).Value = headers(c - 1)
    Next c
    StyleHeader ws, 1, 1, 7

    ' sample rows
    ws.Cells(2, 1).Value = START_DATE: ws.Cells(2, 2).Value = "Amit Sharma": ws.Cells(2, 5).Value = "PROJ-101": ws.Cells(2, 6).Value = 6: ws.Cells(2, 7).Value = "Initial setup"
    ws.Cells(3, 1).Value = START_DATE: ws.Cells(3, 2).Value = "Priya Nair": ws.Cells(3, 5).Value = "PROJ-102": ws.Cells(3, 6).Value = 4: ws.Cells(3, 7).Value = "Test case prep"
    ws.Cells(4, 1).Value = START_DATE: ws.Cells(4, 2).Value = "John Miller": ws.Cells(4, 5).Value = "PROJ-101": ws.Cells(4, 6).Value = 3: ws.Cells(4, 7).Value = "Code review support"

    ' formulas + formatting for all rows 2..LAST_LOG_ROW
    Dim rng As Range
    Set rng = ws.Range("C2:C" & LAST_LOG_ROW)
    rng.Formula = "=IFERROR(INDEX(Resources!$B$2:$B$51,MATCH(B2,Resources!$A$2:$A$51,0)),"""")"

    Set rng = ws.Range("D2:D" & LAST_LOG_ROW)
    rng.Formula = "=IFERROR(INDEX(Resources!$C$2:$C$51,MATCH(B2,Resources!$A$2:$A$51,0)),"""")"

    ws.Range("A2:A" & LAST_LOG_ROW).NumberFormat = "dd-mmm-yyyy"

    ShadeInput ws.Range("A2:B" & LAST_LOG_ROW)
    ShadeInput ws.Range("E2:G" & LAST_LOG_ROW)
    ws.Range("C2:D" & LAST_LOG_ROW).Borders.LineStyle = xlContinuous
    ws.Range("C2:D" & LAST_LOG_ROW).Borders.Color = RGB(191, 191, 191)

    ' dropdowns
    With ws.Range("B2:B" & LAST_LOG_ROW).Validation
        .Delete
        .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, Formula1:="=Resources!$A$2:$A$51"
    End With
    With ws.Range("E2:E" & LAST_LOG_ROW).Validation
        .Delete
        .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, Formula1:="='Jira Tickets'!$A$2:$A$51"
    End With
    With ws.Range("A2:A" & LAST_LOG_ROW).Validation
        .Delete
        .Add Type:=xlValidateDate, AlertStyle:=xlValidAlertStop, Operator:=xlBetween, _
             Formula1:="=DATE(2026,7,3)", Formula2:="=DATE(2026,8,4)"
        .ErrorMessage = "Date must be between 03-Jul-2026 and 04-Aug-2026"
    End With

    ws.Columns("A").ColumnWidth = 13
    ws.Columns("B").ColumnWidth = 18
    ws.Columns("C").ColumnWidth = 14
    ws.Columns("D").ColumnWidth = 10
    ws.Columns("E").ColumnWidth = 14
    ws.Columns("F").ColumnWidth = 14
    ws.Columns("G").ColumnWidth = 30
    ws.Application.ActiveWindow.FreezePanes = False
    ws.Range("A2").Select
    ws.Application.ActiveWindow.FreezePanes = True
End Sub

' ------------------------------------------------------------
Private Sub BuildMatrix(ws As Worksheet)
    ws.Range("A1").Value = "Resource Name"
    ws.Range("B1").Value = "Location"
    ws.Range("C1").Value = "Role"

    Dim numDays As Long
    numDays = DateDiff("d", START_DATE, END_DATE) + 1

    Dim d As Long, col As Long
    For d = 0 To numDays - 1
        col = 4 + d
        ws.Cells(1, col).Value = START_DATE + d
        ws.Cells(1, col).NumberFormat = "dd-mmm"
        ws.Cells(1, col).Orientation = 90
    Next d

    StyleHeader ws, 1, 1, 3 + numDays
    ws.Rows(1).RowHeight = 60

    Dim i As Long, colLetter As String
    For i = 2 To MAX_RES + 1
        ws.Cells(i, 1).Formula = "=IFERROR(INDEX(Resources!$A$2:$A$51,ROW()-1),"""")"
        ws.Cells(i, 2).Formula = "=IFERROR(INDEX(Resources!$B$2:$B$51,ROW()-1),"""")"
        ws.Cells(i, 3).Formula = "=IFERROR(INDEX(Resources!$C$2:$C$51,ROW()-1),"""")"
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 3)).Borders.LineStyle = xlContinuous
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 3)).Borders.Color = RGB(191, 191, 191)
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 3)).Font.Name = "Arial"
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 3)).Font.Size = 10

        For d = 0 To numDays - 1
            col = 4 + d
            colLetter = Split(ws.Cells(1, col).Address(True, False), "$")(0)
            ws.Cells(i, col).Formula = "=IF($A" & i & "="""","""",SUMIFS('Effort Log'!$F$2:$F$" & LAST_LOG_ROW & _
                ",'Effort Log'!$B$2:$B$" & LAST_LOG_ROW & ",$A" & i & _
                ",'Effort Log'!$A$2:$A$" & LAST_LOG_ROW & "," & colLetter & "$1))"
            ws.Cells(i, col).NumberFormat = "0.0;;"""""
            ws.Cells(i, col).HorizontalAlignment = xlCenter
            ws.Cells(i, col).Font.Size = 9
            ws.Cells(i, col).Font.Name = "Arial"
            ws.Cells(i, col).Borders.LineStyle = xlContinuous
            ws.Cells(i, col).Borders.Color = RGB(191, 191, 191)
        Next d
    Next i

    Dim lastColLetter As String
    lastColLetter = Split(ws.Cells(1, 3 + numDays).Address(True, False), "$")(0)
    Dim dataRng As Range
    Set dataRng = ws.Range("D2:" & lastColLetter & (MAX_RES + 1))

    dataRng.FormatConditions.Delete
    With dataRng.FormatConditions.Add(Type:=xlExpression, Formula1:="=AND($A2<>"""",D2=0)")
        .Interior.Color = RGB(248, 203, 173)
    End With
    With dataRng.FormatConditions.Add(Type:=xlExpression, Formula1:="=AND($A2<>"""",D2>0)")
        .Interior.Color = RGB(198, 224, 180)
    End With

    ws.Columns("A").ColumnWidth = 22
    ws.Columns("B").ColumnWidth = 12
    ws.Columns("C").ColumnWidth = 10
    For d = 0 To numDays - 1
        ws.Columns(4 + d).ColumnWidth = 5.5
    Next d

    ws.Cells(MAX_RES + 4, 1).Value = "Legend: Green = hours logged that day. Red = resource exists but logged 0 hours that day. Blank = row unused."
    ws.Cells(MAX_RES + 4, 1).Font.Italic = True
    ws.Cells(MAX_RES + 4, 1).Font.Size = 9
    ws.Cells(MAX_RES + 4, 1).Font.Name = "Arial"

    ws.Application.ActiveWindow.FreezePanes = False
    ws.Range("D2").Select
    ws.Application.ActiveWindow.FreezePanes = True
End Sub

' ------------------------------------------------------------
Private Sub BuildTicketSummary(ws As Worksheet)
    ws.Range("A1").Value = "Jira Ticket"
    ws.Range("B1").Value = "Scope Item / Task Description"
    ws.Range("C1").Value = "Total Effort (Hours)"
    ws.Range("D1").Value = "# Resources Who Logged Time"
    StyleHeader ws, 1, 1, 4

    Dim i As Long
    For i = 2 To MAX_TIX + 1
        ws.Cells(i, 1).Formula = "=IFERROR(INDEX('Jira Tickets'!$A$2:$A$51,ROW()-1),"""")"
        ws.Cells(i, 2).Formula = "=IFERROR(INDEX('Jira Tickets'!$B$2:$B$51,ROW()-1),"""")"
        ws.Cells(i, 3).Formula = "=IF($A" & i & "="""","""",SUMIFS('Effort Log'!$F$2:$F$" & LAST_LOG_ROW & _
            ",'Effort Log'!$E$2:$E$" & LAST_LOG_ROW & ",$A" & i & "))"
        ws.Cells(i, 4).Formula = "=IF($A" & i & "="""","""",SUMPRODUCT(('Effort Log'!$E$2:$E$" & LAST_LOG_ROW & "=$A" & i & _
            ")/COUNTIFS('Effort Log'!$E$2:$E$" & LAST_LOG_ROW & ",'Effort Log'!$E$2:$E$" & LAST_LOG_ROW & "&"""",'Effort Log'!$B$2:$B$" & LAST_LOG_ROW & ",'Effort Log'!$B$2:$B$" & LAST_LOG_ROW & "&""""))"
        ws.Cells(i, 3).NumberFormat = "0.0"
        ws.Cells(i, 4).NumberFormat = "0"
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 4)).Font.Name = "Arial"
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 4)).Font.Size = 10
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 4)).Borders.LineStyle = xlContinuous
        ws.Range(ws.Cells(i, 1), ws.Cells(i, 4)).Borders.Color = RGB(191, 191, 191)
    Next i

    ws.Columns("A").ColumnWidth = 16
    ws.Columns("B").ColumnWidth = 45
    ws.Columns("C").ColumnWidth = 20
    ws.Columns("D").ColumnWidth = 24
    ws.Application.ActiveWindow.FreezePanes = False
    ws.Range("A2").Select
    ws.Application.ActiveWindow.FreezePanes = True
End Sub
