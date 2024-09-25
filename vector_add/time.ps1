$startTime = Get-Date
& ./$args
$endTime = Get-Date
$elapsedTime = New-TimeSpan -Start $startTime -End $endTime
Write-Host "$args time: $elapsedTime seconds"