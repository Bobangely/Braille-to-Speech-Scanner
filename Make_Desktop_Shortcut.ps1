# สร้าง shortcut "สถานีอ่านเบรลล์" บน desktop ชี้ไปที่ Start_Finger_Reader.bat ในโฟลเดอร์นี้
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')  # รองรับ desktop ที่อยู่ใน OneDrive
$link = Join-Path $desktop 'สถานีอ่านเบรลล์.lnk'
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$shortcut.TargetPath = Join-Path $root 'Start_Finger_Reader.bat'
$shortcut.WorkingDirectory = $root
$shortcut.IconLocation = (Join-Path $root 'assets\icon.ico') + ',0'
$shortcut.Description = 'เปิดสถานีอ่านเบรลล์และหน้าเว็บครู'
$shortcut.Save()
Write-Host "สร้างแล้ว: $link"
